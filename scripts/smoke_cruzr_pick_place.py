#!/usr/bin/env python3
# Imported from the local Show-Harness snapshot; see docs/import-manifest.json.
# Modified 2026-11-15: standalone package imports, portable paths, and clarified scope.
"""Deterministic, physics-only Cruzr pick-and-place smoke test.

This validates the MuJoCo robot/IK/gripper stack without a VLM.  It deliberately
uses a pre-grasp trajectory (lift, align, descend) so the open fingers never
push the cube sideways while approaching at table height.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cruzr_sim.mujoco_env import MujocoCruzrRobot, _rot_to_quat
from cruzr_sim.scene import TABLE_Z, TIP_OFFSET, gripper_origin, gripper_tip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plate-x", type=float)
    parser.add_argument("--plate-y", type=float)
    args = parser.parse_args()

    robot = MujocoCruzrRobot(verbose=False, render_size=512)
    model, data = robot.model, robot.data
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    plate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
    if args.plate_x is not None:
        model.body_pos[plate, 0] = args.plate_x
    if args.plate_y is not None:
        model.body_pos[plate, 1] = args.plate_y
    mujoco.mj_forward(model, data)
    for _ in range(240):
        mujoco.mj_step(model, data)

    _, grasp_rot = gripper_origin(model, data, robot.side)
    grasp_quat = _rot_to_quat(grasp_rot)
    stage_steps: dict[str, int] = {}

    def move_tip(goal, label: str, step=0.015, max_iter=100, tol=0.008):
        goal = np.asarray(goal, dtype=float)
        for iteration in range(max_iter):
            current = gripper_tip(model, data, robot.side)
            delta = goal - current
            distance = float(np.linalg.norm(delta))
            if distance < tol:
                stage_steps[label] = iteration
                return True, distance
            delta *= min(1.0, step / distance)
            waypoint = current + delta
            origin = waypoint - grasp_rot[:, 2] * TIP_OFFSET
            robot.update_desired_ee_pose(np.r_[origin, grasp_quat])
        distance = float(np.linalg.norm(goal - gripper_tip(model, data, robot.side)))
        stage_steps[label] = max_iter
        return False, distance

    cube_start = data.xpos[cube].copy()
    plate_pos = data.xpos[plate].copy()
    start_tip = gripper_tip(model, data, robot.side)
    stages = [
        ("prelift", [start_tip[0], start_tip[1], 0.84], 0.015),
        ("prealign", [cube_start[0], cube_start[1], 0.84], 0.018),
        ("descend", [cube_start[0], cube_start[1], 0.753], 0.015),
    ]
    errors = {}
    for label, goal, step in stages:
        ok, error = move_tip(goal, label, step=step)
        errors[label] = error
        if not ok:
            raise RuntimeError(f"{label} failed: {error * 1000:.1f} mm")

    grasp_width = robot.control_gripper(True)
    if grasp_width <= 0.03:
        raise RuntimeError(f"grasp failed: aperture={grasp_width:.4f} m")

    stages = [
        ("lift", [cube_start[0], cube_start[1], 0.84], 0.015),
        ("transfer", [plate_pos[0], plate_pos[1], 0.84], 0.018),
        ("lower", [plate_pos[0], plate_pos[1], 0.755], 0.015),
    ]
    for label, goal, step in stages:
        ok, error = move_tip(goal, label, step=step)
        errors[label] = error
        if not ok:
            raise RuntimeError(f"{label} failed: {error * 1000:.1f} mm")

    robot.control_gripper(False)
    for _ in range(400):
        mujoco.mj_step(model, data)

    cube_pos = data.xpos[cube].copy()
    cube_joint = int(model.body_jntadr[cube])
    cube_dof = int(model.jnt_dofadr[cube_joint])
    cube_speed = float(np.linalg.norm(data.qvel[cube_dof:cube_dof + 3]))
    plate_distance = float(np.linalg.norm(cube_pos[:2] - plate_pos[:2]))
    success = bool(
        plate_distance < 0.06
        and TABLE_Z + 0.01 <= float(cube_pos[2]) <= TABLE_Z + 0.06
        and cube_speed < 0.05
    )

    if args.render_dir:
        import cv2

        args.render_dir.mkdir(parents=True, exist_ok=True)
        front, wrist = robot.camera_frames()
        cv2.imwrite(
            str(args.render_dir / "front.jpg"),
            cv2.cvtColor(front, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(args.render_dir / "wrist.jpg"),
            cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR),
        )

    result = {
        "success": success,
        "cube": cube_pos.round(5).tolist(),
        "plate": plate_pos.round(5).tolist(),
        "plate_xy_distance_m": round(plate_distance, 6),
        "cube_speed_m_s": round(cube_speed, 6),
        "grasp_width_m": round(float(grasp_width), 6),
        "stage_steps": stage_steps,
        "stage_error_mm": {k: round(v * 1000, 2) for k, v in errors.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.verbose else None))
    robot.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
