#!/usr/bin/env python3
"""Load the configured scene and print the named-joint/control contract."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ['MUJOCO_GL'] = 'disable'
from cruzr_sim import CruzrEnvironment


def main():
    env = CruzrEnvironment()
    try:
        obs = env.reset()
        print(f"model: nq={env.model.nq} nv={env.model.nv} nu={env.model.nu}")
        print(f"control: {1 / env.period:.3f} Hz, {env.substeps} physics steps/control")
        print("joints:", ", ".join(obs["joint_names"]))
        print("qpos:", [round(float(x), 6) for x in obs["joint_position"]])
        print("gripper_width_m:", round(obs["gripper_width_m"], 6))
    finally:
        env.close()


if __name__ == "__main__":
    main()
