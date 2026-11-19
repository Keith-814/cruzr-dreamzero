"""Oracle MuJoCo backend for the pick/scan/return workflow, not learned policy."""
import numpy as np
from .ik import ArmIK
from .scene import TIP_OFFSET, gripper_origin, gripper_tip


class SimScanBackend:
    def __init__(self, env, *, images=False, scan_offset=(0., 0., 0.), return_tolerance=0.02):
        self.env, self.images = env, images
        self.ik = ArmIK(env.model, env.data, 'R')
        self.cube_id = env.model.body('cube').id
        self.cube_joint = env.model.body_jntadr[self.cube_id]
        self.table_z = sum((env.config['task']['table_position'][2], env.config['task']['table_half_size'][2]))
        self.half = env.config['task']['cube_half_size']
        self.stopped, self.sequence, self.last_capture = False, 0, None
        self.return_tolerance = return_tolerance
        self.wait(1.)
        self.initial = env.data.xpos[self.cube_id].copy()
        _, current_rotation = gripper_origin(env.model, env.data, 'R')
        xyz = self.initial
        high = self.table_z+0.12
        low = self.table_z+2*self.half+0.003
        offset = np.asarray(scan_offset, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Scan offset requires three finite metres')
        presentation = env.config['task']['presentation']
        yaw = float(presentation['yaw'])
        yaw_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                 [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.rotation = yaw_rotation @ current_rotation
        face = np.asarray(presentation['position'], dtype=float) + offset
        # Fail before the first motion if the configured face pose is outside
        # the arm workspace. The threshold is intentionally conservative and
        # still requires server-side physical validation.
        _, face_pe, face_re = self.ik.solve(face-self.rotation[:, 2]*TIP_OFFSET, self.rotation,
                                            q_seed=env.command.copy(), iters=300, damp=0.1, step=0.5)
        if face_pe > 0.008 or face_re > 0.1:
            raise ValueError(f'Presentation pose is not reachable: {face_pe:.3f} m/{face_re:.3f} rad')
        self.goals = {
            'pregrasp': np.array([xyz[0], xyz[1], high]),
            'grasp': np.array([xyz[0], xyz[1], low]),
            'lift': np.array([xyz[0], xyz[1], high]),
            # Historical key retained for plans/policy phases; this is the
            # explicit face presentation pose, not a table-height lift.
            'scan': face,
            'return_high': np.array([xyz[0], xyz[1], high]),
            'return_low': np.array([xyz[0], xyz[1], low]),
            'retreat': np.array([xyz[0], xyz[1], high]),
        }

    def _check(self, holding):
        if self.stopped:
            raise RuntimeError('Task backend stopped')
        if holding:
            width = self.env.observe()['gripper_width_m']
            if width < self.half or self.env.data.xpos[self.cube_id, 2] < self.table_z-0.02:
                raise RuntimeError('Grasp lost')

    def move(self, waypoint, width, *, holding=False):
        self._check(holding)
        goal = self.goals[waypoint]
        # First lift vertically, before lateral pre-grasp alignment.
        if waypoint == 'pregrasp':
            current = gripper_tip(self.env.model, self.env.data, 'R')
            self._move(np.array([current[0], current[1], goal[2]]), width, False)
        self._move(goal, width, holding)
        if waypoint == 'lift' and self.env.data.xpos[self.cube_id, 2] < self.table_z+self.half+0.03:
            raise RuntimeError('Object did not lift')

    def _move(self, goal, width, holding):
        for _ in range(600):
            self._check(holding)
            tip = gripper_tip(self.env.model, self.env.data, 'R')
            delta = goal-tip
            if np.linalg.norm(delta) < 0.008:
                return
            waypoint = tip+delta*min(1, 0.015/max(np.linalg.norm(delta), 1e-9))
            target, pe, re = self.ik.solve(waypoint-self.rotation[:, 2]*TIP_OFFSET, self.rotation,
                                          iters=220, damp=0.1, step=0.6)
            if pe > 0.03 or re > 0.3:
                raise RuntimeError('Unreachable scan-task waypoint')
            self.env.step(target, width)
        raise RuntimeError('Motion did not converge')

    def grip(self, width, *, grasp=False):
        for _ in range(round(1.2/self.env.period)):
            self.env.step(self.env.command.copy(), width)
        if grasp and self.env.observe()['gripper_width_m'] <= self.half:
            raise RuntimeError('Empty gripper; grasp failed')
        if not grasp and self.env.observe()['gripper_width_m'] < 2*self.half:
            raise RuntimeError('Gripper did not open')

    def wait(self, seconds, *, holding=False):
        for _ in range(round(seconds/self.env.period)):
            self._check(holding)
            self.env.step(self.env.command.copy())

    def capture(self):
        self.sequence += 1
        obs = self.env.observe_pi05(images=self.images)
        self.last_capture = {'id': self.sequence, 'simulation_time': obs['simulation_time'],
                'images': {key.rsplit('.', 1)[-1]: value for key, value in obs.items()
                           if key.startswith('observation.images.')}}
        return self.last_capture

    def verify_return(self):
        pos = self.env.data.xpos[self.cube_id]
        dof = self.env.model.jnt_dofadr[self.cube_joint]
        return bool(np.linalg.norm(pos[:2]-self.initial[:2]) < self.return_tolerance and
                    abs(pos[2]-self.initial[2]) < self.half and
                    np.linalg.norm(self.env.data.qvel[dof:dof+3]) < 0.05 and
                    self.env.observe()['gripper_width_m'] > 2*self.half)

    @property
    def policy_hz(self):
        return self.env.config['policy']['pi05']['sample_hz']

    def policy_bounds(self):
        return self.env.model.jnt_range[self.env.jids].T

    def waypoint_reached(self, waypoint):
        tip = gripper_tip(self.env.model, self.env.data, 'R')
        return bool(np.linalg.norm(tip-self.goals[waypoint]) < .008)

    def policy_observe(self):
        return self.env.observe_pi05(images=True)

    def apply_policy(self, actions, frames, width, *, holding=False):
        # Validate the entire returned chunk, not only the executed prefix.
        actions = self.env._validate_policy_chunk(actions)
        for action in actions[:frames]:
            self._check(holding)
            command = np.r_[action[:7], width]  # Task owns the gripper phase.
            self.env.step_pi05(command)

    def stop(self):
        self.stopped = True  # Physics is caller-driven; no background worker.
