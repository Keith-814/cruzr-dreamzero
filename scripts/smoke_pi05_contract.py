#!/usr/bin/env python3
"""Check the Cruzr reference data/action contract, not Pi0.5 model performance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = args.gl if args.render else 'disable'
    import numpy as np
    from cruzr_sim import CruzrEnvironment

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = None
    result = {"success": False, "model_inference": False, "render_requested": args.render}
    try:
        env = CruzrEnvironment(args.config)
        if not 1 <= args.frames <= env.config["policy"]["pi05"]["action_horizon"]:
            raise ValueError("--frames must lie in [1, configured action_horizon]")
        state = env.observe_pi05(images=False)["observation.state"]
        action = np.r_[state[:7], 0.05]
        chunk = np.repeat(action[None, :], args.frames, axis=0)
        observations = env.step_pi05_chunk(chunk)
        timestamps = np.array([o["simulation_time"] for o in observations])
        expected = args.frames / env.config["policy"]["pi05"]["sample_hz"]
        if abs(timestamps[-1] - expected) > env.model.opt.timestep:
            raise RuntimeError("Policy time drift exceeds one physics timestep")
        result.update(
            frames=args.frames, state_shape=list(state.shape),
            simulation_time=float(timestamps[-1]), expected_time_s=expected,
            policy_hz=env.config["policy"]["pi05"]["sample_hz"],
            physics_dt=env.model.opt.timestep, low_level_control_hz=1/env.period,
            gripper_target_clipped=any(o["gripper_target_clipped"] for o in observations),
            camera_calibrated=False,
        )
        if args.output_dir:
            np.savez_compressed(args.output_dir / "trajectory.npz",
                                simulation_time=timestamps,
                                state=np.stack([o["observation.state"] for o in observations]),
                                requested_action=chunk,
                                applied_action=np.stack([o["applied_action"] for o in observations]))
        if args.render:
            import cv2
            obs = env.observe_pi05(images=True)
            shapes = {}
            for key in ("head_left", "wrist_left"):
                image = obs[f"observation.images.{key}"]
                c = env.config["camera"]
                prefix = "head" if key == "head_left" else "wrist"
                expected_shape = (c[f"{prefix}_height"], c[f"{prefix}_width"], 3)
                if (image.shape != expected_shape or image.dtype != np.uint8 or
                        not np.any(np.ptp(image, axis=(0, 1)))):
                    raise RuntimeError(f"Invalid/constant RGB image for {key}: {image.shape}")
                shapes[key] = list(image.shape)
                if args.output_dir and not cv2.imwrite(str(args.output_dir / f"{key}.png"),
                                                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Failed to save {key}")
            result["image_shapes"] = shapes
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        result["wall_time_s"] = time.perf_counter() - started
        if args.output_dir:
            (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
