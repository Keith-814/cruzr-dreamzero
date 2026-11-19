"""Robot-specific action validation and simulation-time policy sampling.

No model inference, ROS, or MuJoCo dependency. DreamZero outputs must pass an
explicit embodiment adapter before using this absolute Cruzr joint contract.
"""
import math
import numpy as np


class SampleClock:
    """Quantize cumulative sample times, keeping drift below one physics step."""

    def __init__(self, physics_dt, sample_hz):
        if not (math.isfinite(physics_dt) and physics_dt > 0 and
                math.isfinite(sample_hz) and 0 < sample_hz <= 1/physics_dt):
            raise ValueError("sample_hz must be positive and no higher than physics_hz")
        self.steps_per_sample = 1 / (physics_dt * sample_hz)
        self.index = 0

    def next_steps(self):
        steps = round((self.index + 1) * self.steps_per_sample) - round(self.index * self.steps_per_sample)
        self.index += 1
        return steps


def validate_chunk(actions, lower, upper, *, max_frames=50, gripper_max=0.05):
    """Validate the ENTIRE chunk before executing its first action.

    A small tolerance permits float32 representations of exact physical limits.
    Only rounding at those limits is clamped; substantial joint errors fail.
    """
    chunk = np.array(actions, dtype=float, copy=True)
    if chunk.ndim != 2 or chunk.shape[1] != 8 or not 1 <= len(chunk) <= max_frames:
        raise ValueError(f"action chunk must have shape (N, 8), 1 <= N <= {max_frames}")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains NaN or infinity")
    lo = np.r_[lower, 0.0]
    hi = np.r_[upper, gripper_max]
    if np.any(chunk < lo - 1e-7) or np.any(chunk > hi + 1e-7):
        raise ValueError("action chunk outside Cruzr joint/gripper limits")
    return np.clip(chunk, lo, hi)
