# Imported from the local Show-Harness snapshot; see docs/import-manifest.json.
# Modified 2026-11-15: standalone package imports, portable paths, and clarified scope.
"""Cruzr S2 MuJoCo 场景的常量与初始姿态。

场景约定（与 sim/cruzr_s2_urdf.xml 一致）：
    机器人机身和头部朝 +Y 桌面，底座原点在 (0, 0.20, 0.02)，
    底盘前部停靠在桌边下方；
    桌面 z=0.72、中心 y=0.72、尺寸 1.0×0.6 m（靠近机器人一侧 y=0.42，避开底盘半径 ~0.35）；
    方块 (0.05, 0.62, 0.75)，盘子 (0.15, 0.58)。

初始姿态为什么这么定：
    * vendor URDF 的前进轴是本地 +X，因此 MJCF 根节点绕 Z 旋转 +90°，
      使机身正面与头部正对世界 +Y 的桌面；
    * 旋转后使用右臂执行任务。HOME_R 由 `tools/solve_cruzr_home.py`
      多初值求解得到；原注释中的误差数值未经本次验证，
      肘关节保持约 80° 的明显弯曲；
    * 左臂保持零位收起，避免与桌面和任务右臂互相干涉。
"""

from __future__ import annotations

import numpy as np

# ---- 场景常量 --------------------------------------------------------------
TABLE_Z = 0.72
TABLE_Y = 0.72
CUBE_XY = (0.05, 0.62)  # 与 MJCF 里 cube 的 pos 对齐
PLATE_XY = (0.15, 0.58)  # 约 11 cm 运输距离，位于弯肘姿态的稳定可达区
BASE_Z = 0.02
TIP_OFFSET = 0.22  # pgc_base 原点到指尖（URDF 手指 joint 的 z 偏移）

ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_roll",
    "elbow_yaw",
    "wrist_pitch",
    "wrist_roll",
)

# ---- 初始姿态（关节角，弧度） ----------------------------------------------
# 左臂：零位收起。
HOME_L = np.zeros(7)
# 右臂：在方块上方的安全预抓取高度，夹爪竖直朝下，指尖 z≈0.840。
HOME_R = np.array([-1.6469613, -0.1120097, -0.4287706, -1.3961223,
                   -0.3983825, 0.9109803, 0.5171013])


def arm_qpos_addr(model, side: str):
    import mujoco

    return np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{c}_joint")]
        for c in ARM_JOINTS
    ])


def reset_home(model, data, sides=("L", "R")):
    """把指定手臂摆到初始姿态，并把速度清零。"""
    for side, q in (("L", HOME_L), ("R", HOME_R)):
        if side not in sides:
            continue
        adr = arm_qpos_addr(model, side)
        data.qpos[adr] = q
    data.qvel[:] = 0.0


def gripper_origin(model, data, side: str):
    """夹爪基座（pgc_base_link）的世界坐标与旋转矩阵。"""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_pgc_base_link")
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def gripper_tip(model, data, side: str):
    """指尖中点的世界坐标：夹爪原点沿局部 +Z 前移 TIP_OFFSET。"""
    pos, rot = gripper_origin(model, data, side)
    return pos + rot[:, 2] * TIP_OFFSET


def fingers(model, data, side: str):
    """两指的实际开口（米），用于 get_gripper_position()。"""
    import mujoco

    total = 0.0
    for k in (1, 2):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger{k}_joint")
        if jid >= 0:
            total += abs(float(data.qpos[model.jnt_qposadr[jid]]))
    return total
