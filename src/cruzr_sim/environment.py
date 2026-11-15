"""Fixed-period, named-joint simulation API. Does not import a real robot SDK."""
import numpy as np
import mujoco
from .builder import build_xml
from .config import load_config
from .scene import ARM_JOINTS


class CruzrEnvironment:
    def __init__(self, config=None):
        self.config = load_config(config)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.names = ["R_" + x + "_joint" for x in ARM_JOINTS]
        self.jids = np.array([self.model.joint(n).id for n in self.names])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.aids = np.array([self.model.actuator("act_" + n).id for n in self.names])
        self.fjids = [self.model.joint(f"R_finger{i}_joint").id for i in (1, 2)]
        self.faids = [self.model.actuator(f"act_R_finger{i}_joint").id for i in (1, 2)]
        self.substeps = round(1 / (self.config["simulation"]["control_hz"] * self.model.opt.timestep))
        self.period = self.substeps * self.model.opt.timestep
        self.renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.config["robot"]["home"]
        g = self.config["gripper"]
        for side in ("L", "R"):
            for index in (1, 2):
                jid = self.model.joint(f"{side}_finger{index}_joint").id
                self.data.qpos[self.model.jnt_qposadr[jid]] = (g["zero_gap"]-g["max_width"])/2
        # Synchronize ALL position actuators, including the inactive arm.
        for aid in range(self.model.nu):
            jid = self.model.actuator_trnid[aid, 0]
            self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.command = self.data.qpos[self.qadr].copy()
        self.width_command = g["max_width"]
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self, images=False):
        width = self.config["gripper"]["zero_gap"] - sum(
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.fjids)
        obs = {"simulation_time": float(self.data.time), "joint_names": list(self.names),
               "joint_position": self.data.qpos[self.qadr].copy(),
               "joint_velocity": self.data.qvel[self.vadr].copy(),
               "gripper_width_m": float(np.clip(width, 0, self.config["gripper"]["max_width"]))}
        if images:
            obs["images"] = self.render()
        return obs

    def step(self, joint_position, gripper_width_m=None):
        target = np.asarray(joint_position, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("joint_position requires seven finite radians in joint_names order")
        lo, hi = self.model.jnt_range[self.jids].T
        if np.any(target < lo) or np.any(target > hi):
            raise ValueError("joint target outside SDK/model bounds")
        width = self.width_command if gripper_width_m is None else float(gripper_width_m)
        if not np.isfinite(width) or not 0 <= width <= self.config["gripper"]["max_width"]:
            raise ValueError("gripper width outside simulation range")
        sim = self.config["simulation"]
        max_delta = sim["max_joint_speed"] * self.period
        next_q = self.command + np.clip(target-self.command, -max_delta, max_delta)
        next_w = self.width_command + np.clip(width-self.width_command,
                   -sim["max_gripper_speed"]*self.period, sim["max_gripper_speed"]*self.period)
        for step in range(1, self.substeps+1):
            alpha = step/self.substeps
            self.data.ctrl[self.aids] = self.command + alpha*(next_q-self.command)
            w = self.width_command + alpha*(next_w-self.width_command)
            self.data.ctrl[self.faids] = (self.config["gripper"]["zero_gap"]-w)/2
            mujoco.mj_step(self.model, self.data)
        self.command, self.width_command = next_q, float(next_w)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite simulation state")
        return self.observe()

    def render(self):
        c = self.config["camera"]
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=c["height"], width=c["width"])
        frames = {}
        for name in ("front_cam", "R_wrist_cam"):
            self.renderer.update_scene(self.data, camera=name)
            frames[name] = self.renderer.render().copy()
        return frames

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
