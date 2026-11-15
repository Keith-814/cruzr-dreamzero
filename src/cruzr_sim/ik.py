# Imported from the local Show-Harness snapshot; see docs/import-manifest.json.
# Modified 2026-11-15: standalone package imports, portable paths, and clarified scope.
"""阻尼最小二乘（DLS）逆运动学，用于 Cruzr S2 的单臂。

MuJoCo 没有内置 IK，而 Show-Harness 的整套控制都建立在"给一个笛卡尔目标位姿"上，
所以这层必须有。真机侧可核实优必选 manipulation_kinematics 接口；这里是仿真
参考实现，未经真机等价性验证，也用于给仿真算初始姿态。

用法（库）：
    ik = ArmIK(model, data, "L")
    q, err = ik.solve(target_pos, target_rot, q_seed)
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_roll",
    "elbow_yaw",
    "wrist_pitch",
    "wrist_roll",
)


class ArmIK:
    def __init__(self, model, data, side: str, tip_offset: float = 0.22):
        import mujoco

        self.mj = mujoco
        self.m = model
        self.d = data
        self.side = side
        self.tip_offset = tip_offset
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_pgc_base_link")
        if self.body < 0:
            raise ValueError(f"body {side}_pgc_base_link not found")
        self.joints = []
        for comp in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{comp}_joint")
            if jid < 0:
                raise ValueError(f"joint {side}_{comp}_joint not found")
            self.joints.append(jid)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.lo = np.array([model.jnt_range[j][0] for j in self.joints])
        self.hi = np.array([model.jnt_range[j][1] for j in self.joints])
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ state
    def fk(self, q=None):
        """当前（或给定）关节角下，夹爪原点的位置与旋转矩阵。"""
        if q is not None:
            self.d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, self.d)
        pos = self.d.xpos[self.body].copy()
        rot = self.d.xmat[self.body].reshape(3, 3).copy()
        return pos, rot

    def tip(self, q=None):
        pos, rot = self.fk(q)
        return pos + rot[:, 2] * self.tip_offset

    # ------------------------------------------------------------------ solve
    def solve(self, target_pos, target_rot, q_seed=None, iters=200, damp=0.08, step=0.5,
              tol=2e-3):
        """Return ``(q, pos_err, rot_err)`` without changing live joint state.

        IK iterations use the shared ``MjData`` as a kinematic scratch space,
        but the arm qpos is restored before returning.  The caller must move to
        the returned target through actuators; otherwise a held object would be
        left behind by an instantaneous qpos jump.
        """
        d, m, mj = self.d, self.m, self.mj
        q_restore = d.qpos[self.qadr].copy()
        if q_seed is not None:
            d.qpos[self.qadr] = np.clip(q_seed, self.lo, self.hi)
        q = d.qpos[self.qadr].copy()

        best_q, best_err = q.copy(), 1e9
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            pos = d.xpos[self.body].copy()
            rot = d.xmat[self.body].reshape(3, 3).copy()
            e_pos = np.asarray(target_pos, dtype=float) - pos
            # 姿态误差：R_err = R_target * R_cur^T -> 轴角
            r_err = np.asarray(target_rot, dtype=float) @ rot.T
            ang = np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1.0, 1.0))
            if ang < 1e-6:
                e_rot = np.zeros(3)
            else:
                axis = np.array(
                    [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
                ) / (2 * np.sin(ang))
                e_rot = axis * ang
            err = np.concatenate([e_pos, 0.35 * e_rot])
            en = np.linalg.norm(err)
            if en < best_err:
                best_err, best_q = en, q.copy()
            if np.linalg.norm(e_pos) < tol and ang < 0.02:
                break

            mj.mj_jac(m, d, self.jacp, self.jacr, pos, self.body)
            J = np.vstack([self.jacp[:, self.dadr], 0.35 * self.jacr[:, self.dadr]])
            # DLS: dq = J^T (J J^T + λ²I)^-1 e
            JJt = J @ J.T + (damp ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            mx = np.abs(dq).max()
            if mx > step:
                dq *= step / mx
            q = np.clip(q + dq, self.lo, self.hi)
            d.qpos[self.qadr] = q

        # Report the residual of the best solution, not merely the last
        # iteration, then restore the physical state used by the simulator.
        d.qpos[self.qadr] = best_q
        mj.mj_kinematics(m, d)
        pos = d.xpos[self.body].copy()
        rot = d.xmat[self.body].reshape(3, 3).copy()
        pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - pos))
        r_err = np.asarray(target_rot, dtype=float) @ rot.T
        rot_err = float(np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1.0, 1.0)))
        d.qpos[self.qadr] = q_restore
        mj.mj_kinematics(m, d)
        return best_q, pos_err, rot_err


def down_rot(yaw: float = 0.0) -> np.ndarray:
    """夹爪朝下（-Z 世界）的姿态：局部 +Z 指向地面。"""
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
