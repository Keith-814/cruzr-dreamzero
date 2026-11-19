"""Validated repository-relative simulation configuration; no device side effects."""
from pathlib import Path
import math
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path or ROOT / "configs/mujoco.toml").resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)

    def number(section, key, *, positive=False, nonnegative=False):
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{section}.{key} must be finite")
        if (positive and value <= 0) or (nonnegative and value < 0):
            raise ValueError(f"invalid {section}.{key}")
        return value

    def vector(section, key, length, *, positive=False, nonnegative=False):
        values = cfg[section][key]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"{section}.{key} requires {length} values")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{section}.{key} requires finite numbers")
            if (positive and value <= 0) or (nonnegative and value < 0):
                raise ValueError(f"invalid {section}.{key}")
        return values

    sim = cfg["simulation"]
    if sim["arm"] != "R":
        raise ValueError("Only the right-arm simulation profile is supported")
    dt = number("simulation", "physics_dt", positive=True)
    hz = number("simulation", "control_hz", positive=True)
    ratio = 1 / (dt * hz)
    if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
        raise ValueError("control period must contain an integer number of physics steps")
    for key in ("max_joint_speed", "max_gripper_speed"):
        number("simulation", key, positive=True)
    for key in ("home", "lower", "upper", "kp", "kv", "force_limit"):
        vector("robot", key, 7, positive=key in ("kp", "kv", "force_limit"))
    robot = cfg["robot"]
    if any(not lo <= home <= hi or lo >= hi for lo, home, hi in
           zip(robot["lower"], robot["home"], robot["upper"])):
        raise ValueError("invalid joint bounds or home outside configured bounds")
    vector("robot", "base_position", 3)
    quat = vector("robot", "base_quaternion_wxyz", 4)
    if not math.isclose(sum(x*x for x in quat), 1, abs_tol=1e-6):
        raise ValueError("base quaternion must have unit norm")
    for key in ("max_width", "zero_gap", "kp", "kv", "force_limit"):
        number("gripper", key, positive=True)
    if cfg["gripper"]["max_width"] > 0.05:
        raise ValueError("gripper.max_width must not exceed 0.05 m")
    # zero_gap is the aperture at finger joint q=0, NOT the maximum aperture.
    # Negative finger coordinates allow a larger opening than zero_gap.
    for key in ("table_position", "cube_position", "target_position"):
        vector("task", key, 3)
    vector("task", "table_half_size", 3, positive=True)
    vector("task", "friction", 3, nonnegative=True)
    for key in ("cube_half_size", "cube_mass", "target_radius"):
        number("task", key, positive=True)
    presentation = cfg["task"].setdefault("presentation", {})
    position = presentation.setdefault("position", [0.05, 1.02, 1.22])
    if (not isinstance(position, list) or len(position) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in position)):
        raise ValueError("task.presentation.position requires three finite metres")
    yaw = presentation.setdefault("yaw", 0.0)
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("task.presentation.yaw must be finite")
    dwell = presentation.setdefault("dwell_s", 1.0)
    if isinstance(dwell, bool) or not isinstance(dwell, (int, float)) or not math.isfinite(dwell) or dwell < 0:
        raise ValueError("task.presentation.dwell_s must be finite and nonnegative")

    camera = cfg["camera"]
    # Older local configs retain their original front/wrist dimensions and pose.
    for key, default in {
        "head_width": camera["width"], "head_height": camera["height"],
        "wrist_width": camera["width"], "wrist_height": camera["height"],
        "head_source": "front_cam", "wrist_source": "R_wrist_cam",
    }.items():
        camera.setdefault(key, default)
    for key in ("width", "height", "head_width", "head_height", "wrist_width", "wrist_height"):
        value = number("camera", key, positive=True)
        if not isinstance(value, int):
            raise ValueError(f"camera.{key} must be an integer")
    for key in ("head_source", "wrist_source"):
        if not isinstance(camera[key], str) or not camera[key]:
            raise ValueError(f"invalid camera.{key}")
    vector("camera", "front_position", 3)
    axes = vector("camera", "front_xyaxes", 6)
    if min(sum(x*x for x in axes[:3]), sum(x*x for x in axes[3:])) < 1e-12:
        raise ValueError("camera axes must be nonzero")
    cross = [axes[1]*axes[5]-axes[2]*axes[4], axes[2]*axes[3]-axes[0]*axes[5],
             axes[0]*axes[4]-axes[1]*axes[3]]
    if sum(x*x for x in cross) < 1e-12:
        raise ValueError("camera axes must not be parallel")
    if not 0 < number("camera", "fovy", positive=True) < 180:
        raise ValueError("camera.fovy must lie in (0, 180)")

    policy = cfg.setdefault("policy", {}).setdefault("pi05", {})
    for key, default in {"sample_hz": 15, "action_horizon": 50,
                         "gripper_open_target": 0.05}.items():
        policy.setdefault(key, default)
    sample_hz = policy["sample_hz"]
    if isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float)) or not math.isfinite(sample_hz) or not 0 < sample_hz <= 1/dt:
        raise ValueError("Pi0.5 sample_hz must lie in (0, physics_hz]")
    horizon = policy["action_horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Pi0.5 action_horizon must be a positive integer")
    if policy["gripper_open_target"] != 0.05:
        raise ValueError("Pi0.5 reference gripper open target must be 0.05 m")
    model = Path(sim["model"]).expanduser()
    # Relative paths are repository-relative, independent of caller's cwd.
    sim["model"] = str(model.resolve() if model.is_absolute() else (ROOT / model).resolve())
    return cfg
