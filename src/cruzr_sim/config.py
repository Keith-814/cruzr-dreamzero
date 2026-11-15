"""Repository-relative MuJoCo configuration; no ROS or network side effects."""
from pathlib import Path
import math
import tomllib

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = Path(path or ROOT / "configs/mujoco.toml").resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    sim = cfg["simulation"]
    if sim["arm"] != "R":
        raise ValueError("This first server profile is validated for R only; add a left-arm SDK profile first")
    dt, hz = float(sim["physics_dt"]), float(sim["control_hz"])
    if not (math.isfinite(dt) and math.isfinite(hz) and dt > 0 and hz > 0):
        raise ValueError("physics_dt and control_hz must be finite and positive")
    ratio = 1 / (dt * hz)
    if round(ratio) < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("control period must contain an integer number of physics steps")
    for key in ("max_joint_speed", "max_gripper_speed"):
        if not math.isfinite(sim[key]) or sim[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("home", "lower", "upper", "kp", "kv", "force_limit"):
        values = cfg["robot"][key]
        if len(values) != 7 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"robot.{key} needs seven finite values")
    if any(lo >= hi for lo, hi in zip(cfg["robot"]["lower"], cfg["robot"]["upper"])):
        raise ValueError("invalid joint bounds")
    if any(v <= 0 for key in ("kp", "kv", "force_limit") for v in cfg["robot"][key]):
        raise ValueError("actuator parameters must be positive")
    if not (0 < cfg["gripper"]["max_width"] <= 0.05):
        raise ValueError("invalid gripper width")
    if min(cfg["camera"]["width"], cfg["camera"]["height"]) < 1:
        raise ValueError("invalid camera resolution")
    model = Path(sim["model"])
    sim["model"] = str(model if model.is_absolute() else ROOT / model)
    return cfg
