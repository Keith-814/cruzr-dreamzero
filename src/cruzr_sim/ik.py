# Imported from the local Show-Harness snapshot; see docs/import-manifest.json.
# Modified 2026-11-15: standalone package imports, portable paths, and clarified scope.
# Modified 2026-11-19: isolate IK iterations from live MuJoCo state and caches.
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


def rotation_vector(rotation):
    """SO(3) log with a stable axis at 180 degrees (Rodrigues is singular)."""
    angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1.0, 1.0))
    skew = np.array([rotation[2, 1]-rotation[1, 2],
                     rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    if angle < 1e-6:
        return skew / 2
    if np.pi-angle < 1e-5:
        # The rotation axis is the eigenvector with eigenvalue +1.
        _, vectors = np.linalg.eigh((rotation+rotation.T)/2)
        axis = vectors[:, -1]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis * angle
    return skew * angle / (2*np.sin(angle))


class ArmIK:
    def __init__(self, model, data, side: str, tip_offset: float = 0.22):
        import mujoco

        self.mj = mujoco
        self.m = model
        self.d = data
        self._scratch = mujoco.MjData(model)
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

        IK iterations use a separate MjData. Live poses, contacts, Jacobians and
        dynamics caches remain untouched, including when a solve raises.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = np.asarray(target_rot, dtype=float)
        if target_pos.shape != (3,) or not np.isfinite(target_pos).all():
            raise ValueError('IK target position requires three finite metres')
        if (target_rot.shape != (3, 3) or not np.isfinite(target_rot).all() or
                not np.allclose(target_rot.T @ target_rot, np.eye(3), atol=1e-6, rtol=0) or
                not np.isclose(np.linalg.det(target_rot), 1, atol=1e-6, rtol=0)):
            raise ValueError('IK target rotation must be a proper orthonormal 3x3 matrix')
        if q_seed is not None:
            q_seed = np.asarray(q_seed, dtype=float)
            if q_seed.shape != (7,) or not np.isfinite(q_seed).all():
                raise ValueError('IK seed requires seven finite joint angles')
        if (isinstance(iters, bool) or not isinstance(iters, int) or iters < 1 or
                any(not np.isfinite(x) or x <= 0 for x in (damp, step, tol))):
            raise ValueError('IK requires positive iterations, damping, step and tolerance')
        d, m, mj = self._scratch, self.m, self.mj
        d.qpos[:] = self.d.qpos
        if q_seed is not None:
            d.qpos[self.qadr] = np.clip(q_seed, self.lo, self.hi)
        q = d.qpos[self.qadr].copy()

        best_q, best_err = q.copy(), 1e9
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            pos = d.xpos[self.body].copy()
            rot = d.xmat[self.body].reshape(3, 3).copy()
            e_pos = target_pos - pos
            # 姿态误差：R_err = R_target * R_cur^T -> 轴角
            r_err = target_rot @ rot.T
            e_rot = rotation_vector(r_err)
            ang = np.linalg.norm(e_rot)
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
        # iteration. All scratch-state changes remain private to the solver.
        d.qpos[self.qadr] = best_q
        mj.mj_kinematics(m, d)
        pos = d.xpos[self.body].copy()
        rot = d.xmat[self.body].reshape(3, 3).copy()
        pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - pos))
        r_err = np.asarray(target_rot, dtype=float) @ rot.T
        rot_err = float(np.arccos(np.clip((np.trace(r_err) - 1) / 2, -1.0, 1.0)))
        return best_q, pos_err, rot_err


def down_rot(yaw: float = 0.0) -> np.ndarray:
    """夹爪朝下（-Z 世界）的姿态：局部 +Z 指向地面。"""
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
