"""Cruzr simulation-time control, explicit policy sampling, and RGB observations."""
import numpy as np
import mujoco
from .builder import build_xml
from .config import load_config
from .scene import ARM_JOINTS
from .policy import SampleClock, validate_chunk


class CruzrEnvironment:
    def __init__(self, config=None):
        self.config = load_config(config)
        self.scene_xml = build_xml(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
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
        self.renderers = {}
        for key in ("head_source", "wrist_source"):
            name = self.config["camera"][key]
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"Missing camera {name!r}; check imported model and camera.{key}")
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.config["robot"]["home"]
        g = self.config["gripper"]
        for side in ("L", "R"):
            for index in (1, 2):
                jid = self.model.joint(f"{side}_finger{index}_joint").id
                self.data.qpos[self.model.jnt_qposadr[jid]] = (g["zero_gap"]-g["max_width"])/2
        for aid in range(self.model.nu):
            jid = self.model.actuator_trnid[aid, 0]
            self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self.command = self.data.qpos[self.qadr].copy()
        self.width_command = g["max_width"]
        self.policy_clock = SampleClock(self.model.opt.timestep, self.config["policy"]["pi05"]["sample_hz"])
        self.faulted = False
        self.max_tracking_error_rad = 0.0
        self.max_measured_joint_speed_rad_s = 0.0
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

    def observe_pi05(self, images=True):
        """Reference keys only: external front view is NOT calibrated head RGB.

        State uses measured simulation joint angles/aperture, never commands.
        SDK pos and this geometric aperture still require hardware calibration.
        """
        obs = self.observe()
        result = {
            "observation.state": np.r_[obs["joint_position"], obs["gripper_width_m"]].astype(np.float32),
            "simulation_time": obs["simulation_time"],
            "joint_names": list(self.names),
            "camera_calibrated": False,
        }
        if images:
            c = self.config["camera"]
            result["observation.images.head_left"] = self._render(c["head_source"], c["head_height"], c["head_width"])
            # Historical Pi0.5 name refers to the physical RIGHT wrist.
            result["observation.images.wrist_left"] = self._render(c["wrist_source"], c["wrist_height"], c["wrist_width"])
        return result

    def _validate_policy_chunk(self, actions):
        lo, hi = self.model.jnt_range[self.jids].T
        return validate_chunk(actions, lo, hi,
                              max_frames=self.config["policy"]["pi05"]["action_horizon"],
                              gripper_max=0.05)

    def step_pi05(self, action):
        """Advance one sample at configured 15 Hz, not one 50 Hz control tick."""
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError("Pi0.5 action requires shape (8,): 7 radians + gripper metres")
        action = self._validate_policy_chunk(action[None, :])[0]
        if self.faulted:
            raise RuntimeError("Simulation is faulted; reset before continuing")
        width = min(float(action[7]), self.config["gripper"]["max_width"])
        self._advance(action[:7], width, self.policy_clock.next_steps())
        result = self.observe_pi05(images=False)
        result["requested_action"] = action.copy()
        result["applied_action"] = np.r_[self.command, self.width_command]
        result["gripper_target_clipped"] = bool(action[7] > width)
        return result

    def step_pi05_chunk(self, actions, *, observe=True):
        """Prevalidate all (N,8) actions, then execute at configured sample_hz.

        This is synchronous simulation time: physics pauses during inference.
        No claim is made about asynchronous real-robot execution or latency.
        """
        chunk = self._validate_policy_chunk(actions)
        observations = []
        for action in chunk:
            observation = self.step_pi05(action)
            if observe:
                observations.append(observation)
        return observations if observe else None

    def step(self, joint_position, gripper_width_m=None):
        """Advance exactly one fixed low-level control period (default 20 ms)."""
        target = np.asarray(joint_position, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("joint_position requires seven finite radians in joint_names order")
        lo, hi = self.model.jnt_range[self.jids].T
        if np.any(target < lo) or np.any(target > hi):
            raise ValueError("joint target outside SDK/model bounds")
        width = self.width_command if gripper_width_m is None else float(gripper_width_m)
        if not np.isfinite(width) or not 0 <= width <= self.config["gripper"]["max_width"]:
            raise ValueError("gripper width outside simulation range")
        self._advance(target, width, self.substeps)
        return self.observe()

    def _advance(self, target, width, steps):
        if self.faulted:
            raise RuntimeError("Simulation is faulted; reset before continuing")
        sim = self.config["simulation"]
        duration = steps * self.model.opt.timestep
        next_q = self.command + np.clip(target-self.command,
                                      -sim["max_joint_speed"]*duration, sim["max_joint_speed"]*duration)
        next_w = self.width_command + np.clip(width-self.width_command,
                   -sim["max_gripper_speed"]*duration, sim["max_gripper_speed"]*duration)
        warnings_before = self.data.warning.number.copy()
        try:
            for index in range(1, steps+1):
                alpha = index/steps
                self.data.ctrl[self.aids] = self.command + alpha*(next_q-self.command)
                w = self.width_command + alpha*(next_w-self.width_command)
                self.data.ctrl[self.faids] = (self.config["gripper"]["zero_gap"]-w)/2
                before = float(self.data.time)
                mujoco.mj_step(self.model, self.data)
                # MuJoCo may reset an unstable state to finite values.
                if not np.isclose(self.data.time-before, self.model.opt.timestep, rtol=0, atol=1e-9):
                    raise RuntimeError("MuJoCo time discontinuity; reset required")
                self._check_physics(warnings_before)
                self.max_tracking_error_rad = max(self.max_tracking_error_rad,
                    float(np.max(np.abs(self.data.qpos[self.qadr]-self.data.ctrl[self.aids]))))
                self.max_measured_joint_speed_rad_s = max(self.max_measured_joint_speed_rad_s,
                    float(np.max(np.abs(self.data.qvel[self.vadr]))))
            # mj_step integrates qpos AFTER computing derived poses. Refresh
            # poses/contacts so images, scoring and joint state share one time.
            mujoco.mj_forward(self.model, self.data)
            self._check_physics(warnings_before)
        except Exception:
            # A native error or user callback can interrupt a partially applied
            # command. Continuing from the old command would be inconsistent.
            self.faulted = True
            raise
        self.command, self.width_command = next_q, float(next_w)

    def _check_physics(self, warnings_before):
        if (np.any(self.data.warning.number > warnings_before) or
                not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc))):
            raise RuntimeError("MuJoCo warning/unstable state; reset required")

    def _render(self, name, height, width):
        key = (name, height, width)
        renderer = self.renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.renderers[key] = renderer
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def render(self):
        """Preserve the original diagnostic camera keys and preview sizes."""
        c = self.config["camera"]
        return {name: self._render(name, c["height"], c["width"])
                for name in ("front_cam", "R_wrist_cam")}

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
