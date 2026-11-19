"""Feedback-controlled hardware executor; no implicit ROS or MuJoCo imports.

The transport must supply monotonic receipt times, motor-supervisor health,
measured SDK joint state, and gripper status. A separate publisher loop enforces
rate, feedback freshness, tracking bounds and a short command lease.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
from .scene import ARM_JOINTS
from .task import WAYPOINTS

JOINT_NAMES = tuple(f'R_{name}_joint' for name in ARM_JOINTS)
SDK_LOWER = np.array([-2.83, -1.86, -2.92, -2.60, -2.88, -1.60, -1.98])
SDK_UPPER = np.array([2.83, 0.08, 2.92, 0.02, 2.88, 1.57, 1.98])


def finite_vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{label} requires {length} finite numbers')
    return result.copy()


def load_profile(path, *, commissioned=False, require_scanner=False):
    profile = json.loads(Path(path).read_text())
    if profile['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Hardware profile joint order mismatch')
    lo = finite_vector(profile['lower'], 7, 'lower')
    hi = finite_vector(profile['upper'], 7, 'upper')
    if np.any(lo < SDK_LOWER) or np.any(hi > SDK_UPPER) or np.any(lo >= hi):
        raise ValueError('Hardware limits must be within right-arm SDK bounds')
    for key in ('command_hz', 'max_joint_speed', 'max_gripper_speed', 'max_state_age_s',
                'max_tracking_error_rad', 'joint_tolerance_rad', 'motion_timeout_s',
                'lease_s', 'gripper_force_n', 'gripper_current_a', 'max_image_age_s'):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid hardware {key}')
    if not 10 <= profile['command_hz'] <= 500 or profile['max_joint_speed'] > 0.2:
        raise ValueError('Commissioning profile supports 10..500 Hz and <=0.2 rad/s targets')
    if profile['max_gripper_speed'] > 0.02 or profile['gripper_force_n'] > 50 or profile['gripper_current_a'] > 0.4:
        raise ValueError('Commissioning gripper envelope exceeded')
    g = profile['gripper_calibration']
    if not 0 < g['geometric_max_m'] <= 0.05:
        raise ValueError('Invalid calibrated geometric aperture')
    if not all(math.isfinite(g[k]) and 0 <= g[k] <= 0.05 for k in ('sdk_open_m', 'sdk_closed_m')) or g['sdk_open_m'] == g['sdk_closed_m']:
        raise ValueError('Invalid gripper SDK calibration endpoints')
    if commissioned:
        checks = profile['commissioning']
        required = ('joint_zero_verified', 'gripper_mapping_verified', 'camera_verified',
                    'sdk_route_verified', 'sparse_command_verified', 'independent_stop_verified',
                    'command_timeout_verified', 'supervisor_verified')
        if require_scanner:
            required += ('scanner_verified',)
        if not checks.get('calibration_id') or any(checks.get(k) is not True for k in required):
            raise ValueError('Hardware commissioning record is incomplete')
        for key in ('health_topic', 'stop_request_topic', 'head_image_topic', 'wrist_image_topic'):
            if not profile['topics'].get(key):
                raise ValueError(f'Commissioned profile requires topics.{key}')
    return profile


def load_plan(path, profile):
    plan = json.loads(Path(path).read_text())
    if plan['joint_names'] != list(JOINT_NAMES):
        raise ValueError('Taught plan joint order mismatch')
    missing = set(('start', *WAYPOINTS)) - set(plan.get('waypoints', {}))
    if missing:
        raise ValueError(f'Taught plan is missing waypoints: {sorted(missing)}; record/review a new plan')
    for name in ('start', *WAYPOINTS):
        q = finite_vector(plan['waypoints'][name], 7, name)
        if np.any(q < profile['lower']) or np.any(q > profile['upper']):
            raise ValueError(f'Taught waypoint outside limits: {name}')
    if plan.get('calibration_id') != profile['commissioning']['calibration_id']:
        raise ValueError('Plan/profile calibration id mismatch')
    return plan


def check_start_state(state, plan, profile):
    """The approved path starts here; do not approach it from an arbitrary pose."""
    measured = finite_vector(state['joint_position'], 7, 'start feedback')
    velocity = finite_vector(state['joint_velocity'], 7, 'start velocity')
    if np.max(np.abs(measured-np.asarray(plan['waypoints']['start']))) > profile['joint_tolerance_rad']:
        raise RuntimeError('Robot is not at the taught start pose; position externally before execute')
    if np.max(np.abs(velocity)) > 0.03:
        raise RuntimeError('Robot must be stationary at the taught start pose')


class HardwareController:
    def __init__(self, transport, profile, *, enabled=False):
        self.transport, self.p = transport, profile
        self.enabled = enabled
        self.lock, self.done = threading.Lock(), threading.Event()
        self.error, self.worker = None, None
        self.q_goal = self.q_command = None
        self.g_goal = self.g_command = None
        self.holding, self.lease = False, 0.

    def state(self):
        if getattr(self.transport, 'error', None):
            raise RuntimeError(f'Transport failure: {self.transport.error}')
        s = self.transport.read_state()
        now = time.monotonic()
        for key in ('arm_received_s', 'gripper_received_s', 'health_received_s'):
            stamp = s.get(key)
            if stamp is None or not math.isfinite(stamp) or not 0 <= now-stamp <= self.p['max_state_age_s']:
                raise RuntimeError(f'Stale or missing {key}')
        if s.get('healthy') is not True:
            raise RuntimeError('Independent robot supervisor is not healthy')
        q = finite_vector(s['joint_position'], 7, 'measured joints')
        v = finite_vector(s['joint_velocity'], 7, 'measured joint velocities')
        if np.any(q < np.asarray(self.p['lower'])-1e-4) or np.any(q > np.asarray(self.p['upper'])+1e-4):
            raise RuntimeError('Measured joints outside profile bounds')
        g = s['gripper']
        if (g['init_state'] != 1 or g['homed'] != 1 or g['error_code'] != 0 or
                g['grip_state'] not in (0, 1, 2) or not math.isfinite(g['pos']) or not 0 <= g['pos'] <= 0.05):
            raise RuntimeError('Gripper unready, dropped, or invalid')
        s['joint_position'], s['joint_velocity'] = q, v
        return s

    def start(self):
        if not self.enabled:
            raise RuntimeError('Read-only controller cannot start a publisher')
        if self.worker is not None:
            raise RuntimeError('Controller cannot be restarted; create a new session')
        s = self.state()
        self.q_goal = self.q_command = s['joint_position'].copy()
        self.g_goal = self.g_command = float(s['gripper']['pos'])
        self.lease = time.monotonic()
        self.worker = threading.Thread(target=self._loop, name='cruzr-command', daemon=True)
        self.worker.start()

    def _loop(self):
        period = 1/self.p['command_hz']
        next_tick = time.monotonic()
        try:
            while not self.done.is_set():
                with self.lock:
                    now = time.monotonic()
                    if now-self.lease > self.p['lease_s']:
                        raise RuntimeError('Command lease expired; queued motion cancelled')
                    s = self.state()
                    if self.holding and s['gripper']['grip_state'] != 2:
                        raise RuntimeError('Grasp feedback lost')
                    if np.max(np.abs(s['joint_position']-self.q_command)) > self.p['max_tracking_error_rad']:
                        raise RuntimeError('Joint tracking error exceeded')
                    # Use one nominal tick even after a scheduler delay; never
                    # catch up with a burst or a large wall-time-based jump.
                    self.q_command = self.q_command + np.clip(self.q_goal-self.q_command,
                        -self.p['max_joint_speed']*period, self.p['max_joint_speed']*period)
                    self.g_command += float(np.clip(self.g_goal-self.g_command,
                        -self.p['max_gripper_speed']*period, self.p['max_gripper_speed']*period))
                    self.transport.publish(self.q_command.copy(), self.g_command)
                next_tick = max(next_tick+period, time.monotonic())
                self.done.wait(max(0, next_tick-time.monotonic()))
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.done.set()
            try:
                self.transport.stop()
            except Exception as stop_exc:
                self.error += f'; stop request failed: {stop_exc}'

    def target(self, q, geometric_width, *, holding=False):
        q = finite_vector(q, 7, 'command target')
        if np.any(q < self.p['lower']) or np.any(q > self.p['upper']):
            raise ValueError('Command target outside profile bounds')
        g = self.p['gripper_calibration']
        if not math.isfinite(geometric_width) or not 0 <= geometric_width <= g['geometric_max_m']:
            raise ValueError('Target aperture outside calibrated range')
        sdk = g['sdk_closed_m']+(g['sdk_open_m']-g['sdk_closed_m'])*geometric_width/g['geometric_max_m']
        with self.lock:
            if not self.enabled or self.worker is None or self.done.is_set():
                raise RuntimeError(self.error or 'Controller not running')
            self.q_goal, self.g_goal, self.holding = q, sdk, holding
            self.lease = time.monotonic()

    def tick(self):
        with self.lock:
            if self.done.is_set():
                raise RuntimeError(self.error or 'Controller stopped')
            self.lease = time.monotonic()
        return self.state()

    def stop(self):
        self.done.set()
        if self.worker:
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                raise RuntimeError('Publisher thread did not terminate')
        self.transport.stop()

    def observe_policy(self):
        s = self.tick()
        capture = self.transport.capture_images()
        return {'observation.state': np.r_[s['joint_position'], s['gripper']['pos']].astype(np.float32),
                **{f'observation.images.{name}': frame for name, frame in capture['images'].items()},
                'monotonic_time': time.monotonic(), 'capture_id': capture['id'],
                'gripper_representation': 'measured_sdk_position_m'}

    def execute_chunk(self, actions, *, sample_hz=15, execute_frames=3, gripper_width, holding=False):
        """Execute a short prefix of de-normalized absolute joint predictions.

        Whole-chunk validation precedes execution. Gripper timing belongs to
        the task state machine, so the eighth predicted value is NOT sent.
        The independent publisher continues feedback checks during inference;
        an inference longer than lease_s cancels motion, requiring a new session.
        """
        from .policy import validate_chunk
        chunk = validate_chunk(actions, self.p['lower'], self.p['upper'])
        if (not math.isfinite(sample_hz) or not 1 <= sample_hz <= self.p['command_hz'] or
                isinstance(execute_frames, bool) or not isinstance(execute_frames, int) or not 1 <= execute_frames <= 5):
            raise ValueError('Invalid policy sample rate/prefix length')
        for action in chunk[:execute_frames]:
            self.target(action[:7], gripper_width, holding=holding)
            deadline = time.monotonic()+1/sample_hz
            while time.monotonic() < deadline:
                self.tick()
                self.done.wait(min(.02, max(0, deadline-time.monotonic())))


class HardwareScanBackend:
    def __init__(self, controller, plan, *, placement_verifier=None):
        self.c, self.plan = controller, plan
        self.placement_verifier = placement_verifier

    def _until(self, predicate):
        deadline, stable = time.monotonic()+self.c.p['motion_timeout_s'], None
        while time.monotonic() < deadline:
            s = self.c.tick()
            if predicate(s):
                stable = stable or time.monotonic()
                if time.monotonic()-stable >= 0.2:
                    return
            else:
                stable = None
            time.sleep(0.02)
        raise TimeoutError('Measured hardware target did not settle')

    def move(self, waypoint, width, *, holding=False):
        q = np.asarray(self.plan['waypoints'][waypoint])
        self.c.target(q, width, holding=holding)
        self._until(lambda s: np.max(np.abs(self.c.q_command-q)) < self.c.p['joint_tolerance_rad']
                    and np.max(np.abs(s['joint_position']-q)) < self.c.p['joint_tolerance_rad']
                    and np.max(np.abs(s['joint_velocity'])) < 0.03)

    def grip(self, width, *, grasp=False):
        self.c.target(self.c.q_goal.copy(), width, holding=False)
        if grasp:
            self._until(lambda s: s['gripper']['grip_state'] == 2)
            self.c.target(self.c.q_goal.copy(), width, holding=True)
        else:
            goal = self.c.g_goal
            self._until(lambda s: s['gripper']['grip_state'] == 1
                        and abs(self.c.g_command-goal) < 0.002 and abs(s['gripper']['pos']-goal) < 0.002)

    def wait(self, seconds, *, holding=False):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            s = self.c.tick()
            if holding and s['gripper']['grip_state'] != 2:
                raise RuntimeError('Object no longer held')
            time.sleep(0.02)

    def capture(self):
        self.c.tick()
        capture = self.c.transport.capture_images()
        self.c.tick()
        return capture

    def verify_return(self):
        # Gripper opening and a completed trajectory do NOT prove placement.
        return self.placement_verifier() if self.placement_verifier else None

    def stop(self):
        self.c.stop()
