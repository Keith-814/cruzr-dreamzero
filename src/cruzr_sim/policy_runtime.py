"""Embodiment-checked policy plugin and phase-supervised motion wrapper.

Plugins must denormalize and undo relative-action transforms BEFORE returning
absolute SDK-order targets. The task owns gripper timing, scanning and release;
model predictions cannot open the gripper during transport of an object.
"""
import json
import math
from pathlib import Path
from .real_robot import JOINT_NAMES
from .policy import validate_chunk

INSTRUCTION = 'Pick up the object, hold it in front of the robot face for inspection, and return it to its original position.'


class PolicyMotionBackend:
    def __init__(self, backend, predictor, manifest, *, execute_frames=3, max_replans=80):
        self.backend, self.predictor = backend, predictor
        self.manifest = json.loads(Path(manifest).read_text())
        m = self.manifest
        if (m.get('embodiment') != 'cruzr_s2' or m.get('joint_names') != list(JOINT_NAMES)+['right_gripper'] or
                m.get('action_semantics') != 'absolute_joint_position' or m.get('joint_unit') != 'rad' or
                m.get('gripper_unit') != 'm' or m.get('denormalized') is not True or not m.get('checkpoint_id')):
            raise ValueError('Policy must declare denormalized Cruzr absolute rad/m targets and checkpoint identity')
        if not math.isclose(m['sample_hz'], backend.policy_hz, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Policy and execution sample rates differ')
        if not 1 <= execute_frames <= 5 or not 1 <= max_replans <= 500:
            raise ValueError('Invalid replanning budget')
        self.frames, self.max_replans = execute_frames, max_replans
        predictor.reset()  # New task means a new model session/cache.

    def move(self, waypoint, width, *, holding=False):
        for _ in range(self.max_replans):
            if self.backend.waypoint_reached(waypoint):
                return
            observation = self.backend.policy_observe()
            action = self.predictor.predict(observation, instruction=INSTRUCTION, phase=waypoint)
            lo, hi = self.backend.policy_bounds()
            action = validate_chunk(action, lo, hi, max_frames=50, gripper_max=.05)
            self.backend.apply_policy(action, min(self.frames, len(action)), width, holding=holding)
        raise TimeoutError(f'Policy failed to reach phase {waypoint}')

    def __getattr__(self, name):
        return getattr(self.backend, name)
