#!/usr/bin/env python3
"""Server headless runner for configured MuJoCo; no model inference or ROS."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path)
parser.add_argument("--mode", choices=["hold", "pick-place"], default="hold")
parser.add_argument("--seconds", type=float, default=2)
parser.add_argument("--render-dir", type=Path)
parser.add_argument("--gl", choices=["egl", "osmesa", "glfw"], default="egl")
args = parser.parse_args()
os.environ["MUJOCO_GL"] = args.gl if args.render_dir else 'disable'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from cruzr_sim import CruzrEnvironment
from cruzr_sim.ik import ArmIK
from cruzr_sim.scene import gripper_origin, gripper_tip, TIP_OFFSET


def main():
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        raise ValueError("--seconds must be positive and finite")
    env = CruzrEnvironment(args.config)
    started = time.perf_counter()
    try:
        home = env.command.copy()
        for _ in range(max(1, round(args.seconds / env.period))):
            env.step(home)
        result = {"mode": args.mode, "control_hz": 1/env.period,
                  "scene_sha256": hashlib.sha256(env.scene_xml.encode()).hexdigest(),
                  "camera_calibrated": False, "model_inference": False}
        if args.mode == "pick-place":
            # Oracle positions from physics are for controller validation only.
            ik = ArmIK(env.model, env.data, "R")
            _, rotation = gripper_origin(env.model, env.data, "R")
            cube_id = env.model.body("cube").id
            cube = env.data.xpos[cube_id].copy()
            plate = env.data.xpos[env.model.body("plate").id].copy()
            table_z = env.config["task"]["table_position"][2] + env.config["task"]["table_half_size"][2]
            half = env.config["task"]["cube_half_size"]
            high = table_z + 0.12
            stages = {}

            def move(goal, label, width):
                goal = np.asarray(goal)
                for iteration in range(600):
                    tip = gripper_tip(env.model, env.data, "R")
                    delta = goal-tip
                    if np.linalg.norm(delta) < 0.008:
                        stages[label] = iteration
                        return
                    waypoint = tip+delta*min(1, 0.015/max(np.linalg.norm(delta), 1e-9))
                    target, pe, re = ik.solve(waypoint-rotation[:, 2]*TIP_OFFSET, rotation,
                                              iters=220, damp=0.1, step=0.6)
                    if pe > 0.03 or re > 0.3:
                        raise RuntimeError(f"{label}: unreachable waypoint, IK residual={pe:.4f}m/{re:.4f}rad")
                    env.step(target, width)
                raise RuntimeError(f"{label}: did not converge")

            opened = env.config["gripper"]["max_width"]
            tip = gripper_tip(env.model, env.data, "R")
            move([tip[0], tip[1], high], "prelift", opened)
            move([cube[0], cube[1], high], "prealign", opened)
            move([cube[0], cube[1], table_z+2*half+0.003], "descend", opened)
            for _ in range(round(0.8/env.period)):
                env.step(env.command.copy(), 0)
            move([cube[0], cube[1], high], "lift", 0)
            if env.data.xpos[cube_id, 2] < table_z+half+0.03:
                raise RuntimeError("Cube was not lifted; width alone is not proof of grasp")
            move([plate[0], plate[1], high], "transfer", 0)
            move([plate[0], plate[1], table_z+2*half+0.005], "lower", 0)
            for _ in range(round(1.2/env.period)):
                env.step(env.command.copy(), opened)
            # A cube supported by fingers near the target is not a placement.
            # Retreat and settle before scoring, so release is part of success.
            move([plate[0], plate[1], high], "retreat", opened)
            for _ in range(round(0.6/env.period)):
                env.step(env.command.copy(), opened)
            pos = env.data.xpos[cube_id].copy()
            jid = env.model.body_jntadr[cube_id]
            dof = env.model.jnt_dofadr[jid]
            speed = np.linalg.norm(env.data.qvel[dof:dof+3])
            distance = np.linalg.norm(pos[:2]-plate[:2])
            success = bool(distance < env.config["task"]["target_radius"] and
                           table_z+half/2 < pos[2] < table_z+3*half and speed < 0.05 and
                           env.observe()["gripper_width_m"] > 2*half)
            result.update(success=success, oracle_positions=True, stage_steps=stages,
                          target_xy_distance_m=float(distance), cube_speed_m_s=float(speed),
                          cube_position=pos.tolist())
        else:
            result["success"] = bool(np.max(np.abs(env.data.qpos[env.qadr]-home)) < 0.05)
        result["simulation_time"] = float(env.data.time)
        result["wall_time_s_without_render"] = time.perf_counter()-started
        result["physics_time_factor"] = result["simulation_time"]/result["wall_time_s_without_render"]
        result["max_tracking_error_rad"] = env.max_tracking_error_rad
        result["max_measured_joint_speed_rad_s"] = env.max_measured_joint_speed_rad_s
        result["mujoco_warning_counts"] = env.data.warning.number.tolist()
        if args.render_dir:
            import cv2
            args.render_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in env.render().items():
                if not cv2.imwrite(str(args.render_dir / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Failed writing {name}")
            (args.render_dir/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
