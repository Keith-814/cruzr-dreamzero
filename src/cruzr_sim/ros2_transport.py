"""Optional SDK ROS2 adapter. Constructing it with writable=False is read-only.

Only message definitions documented in the supplied SDK are used. The health
and stop-request topics are an integration contract for an INDEPENDENT tested
supervisor, not vendor-provided topics. No homing, init or motor reset commands.
"""
import threading
import time
import numpy as np
from .real_robot import JOINT_NAMES


class Ros2Transport:
    def __init__(self, profile, *, writable=False):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from mc_state_msgs.msg import RobotState
        from ecat_task_msgs.msg import GripStatus
        from sensor_msgs.msg import Image
        from std_msgs.msg import Bool
        self.rclpy, self.profile, self.writable = rclpy, profile, writable
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('cruzr_workflow')
        self.lock, self.data, self.frames, self.last_stamps = threading.Lock(), {}, {}, {}
        self.stopped = False
        self.error = None
        topics = profile['topics']
        self.node.create_subscription(RobotState, topics['arm_state_topic'], self._arm, qos_profile_sensor_data)
        self.node.create_subscription(GripStatus, topics['gripper_state_topic'], self._grip, qos_profile_sensor_data)
        if topics['health_topic']:
            self.node.create_subscription(Bool, topics['health_topic'], self._health, qos_profile_sensor_data)
        for name, key in (('head_left', 'head_image_topic'), ('wrist_left', 'wrist_image_topic')):
            if topics[key]:
                self.node.create_subscription(Image, topics[key], lambda msg, n=name: self._image(n, msg), qos_profile_sensor_data)
        self.arm_pub = self.grip_pub = self.stop_pub = None
        if writable:
            from mc_task_msgs.msg import RobotCommand, JointCmd
            from ecat_task_msgs.msg import GripCmd
            from std_msgs.msg import Empty
            self.RobotCommand, self.JointCmd, self.GripCmd, self.Empty = RobotCommand, JointCmd, GripCmd, Empty
            self.arm_pub = self.node.create_publisher(RobotCommand, topics['arm_command_topic'], 1)
            self.grip_pub = self.node.create_publisher(GripCmd, topics['gripper_command_topic'], 1)
            self.stop_pub = self.node.create_publisher(Empty, topics['stop_request_topic'], 1)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            while self.rclpy.ok() and not self.stopped:
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
        except Exception as exc:
            self.error = str(exc)

    def _stamp(self, key, msg):
        stamp = msg.header.stamp
        ns = stamp.sec*1000000000+stamp.nanosec
        age = (self.node.get_clock().now().nanoseconds-ns)/1e9
        if ns <= self.last_stamps.get(key, 0) or not 0 <= age <= self.profile['max_state_age_s']:
            raise RuntimeError(f'{key}: repeated/out-of-order/stale source timestamp; synchronize ROS clocks')
        self.last_stamps[key] = ns
        return ns

    def _arm(self, msg):
        try:
            self._stamp('arm', msg)
            js = msg.joint_states
            if len(js.name) != len(js.position) or len(js.name) != len(js.velocity) or len(set(js.name)) != len(js.name):
                raise ValueError('SDK named joint state length/uniqueness error')
            indices = {name: i for i, name in enumerate(js.name)}
            with self.lock:
                self.data.update(joint_position=[js.position[indices[n]] for n in JOINT_NAMES],
                                 joint_velocity=[js.velocity[indices[n]] for n in JOINT_NAMES], arm_received_s=time.monotonic())
        except Exception as exc:
            self.error = str(exc)
            raise

    def _grip(self, msg):
        try:
            self._stamp('gripper', msg)
            with self.lock:
                self.data.update(gripper={key: getattr(msg, key) for key in
                                 ('init_state', 'homed', 'grip_state', 'error_code', 'pos')}, gripper_received_s=time.monotonic())
        except Exception as exc:
            self.error = str(exc)
            raise

    def _health(self, msg):
        with self.lock:
            self.data.update(healthy=bool(msg.data), health_received_s=time.monotonic())

    def _image(self, name, msg):
        try:
            # Image clocks use the same ROS clock, with a separate camera age limit.
            ns = msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
            age = (self.node.get_clock().now().nanoseconds-ns)/1e9
            if ns <= self.last_stamps.get(name, 0) or not 0 <= age <= self.profile['max_image_age_s']:
                return
            if msg.encoding not in ('rgb8', 'bgr8') or msg.step < msg.width*3:
                raise ValueError('Use decoded rgb8/bgr8 sensor_msgs/Image topics')
            pixels = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
            image = pixels[:, :msg.width*3].reshape(msg.height, msg.width, 3)
            if msg.encoding == 'bgr8':
                image = image[:, :, ::-1]
            with self.lock:
                self.frames[name] = (ns, time.monotonic(), image.copy())
                self.last_stamps[name] = ns
        except Exception as exc:
            self.error = str(exc)
            raise

    def read_state(self):
        if self.error:
            raise RuntimeError(f'ROS subscription failure: {self.error}')
        with self.lock:
            return dict(self.data)

    def capture_images(self):
        with self.lock:
            frames = dict(self.frames)
        if set(frames) != {'head_left', 'wrist_left'}:
            raise RuntimeError('Both image streams are required')
        if any(time.monotonic()-row[1] > self.profile['max_image_age_s'] for row in frames.values()):
            raise RuntimeError('Camera frames are stale')
        stamps = [row[0] for row in frames.values()]
        if (max(stamps)-min(stamps))/1e9 > self.profile['max_image_age_s']:
            raise RuntimeError('Camera streams not synchronized')
        return {'id': '-'.join(str(frames[key][0]) for key in sorted(frames)),
                'source_stamp_ns': {key: value[0] for key, value in frames.items()},
                'images': {key: value[2] for key, value in frames.items()}}

    def publish(self, q, sdk_gripper):
        if not self.writable or self.stopped:
            raise RuntimeError('Transport is read-only/stopped')
        cmd = self.RobotCommand()
        cmd.header.stamp = self.node.get_clock().now().to_msg()
        for name, position in zip(JOINT_NAMES, q):
            joint = self.JointCmd()
            joint.name, joint.control_mode, joint.position = name, 2, float(position)
            cmd.joint_cmd.append(joint)
        grip = self.GripCmd()
        now = self.node.get_clock().now().to_msg()
        grip.header.stamp.sec, grip.header.stamp.nanosec = now.sec, now.nanosec
        grip.pos, grip.vel = float(sdk_gripper), self.profile['max_gripper_speed']
        grip.force, grip.cur, grip.mode = self.profile['gripper_force_n'], self.profile['gripper_current_a'], 0
        # init/reset/homing remain zero; commissioning is performed externally.
        self.arm_pub.publish(cmd)
        self.grip_pub.publish(grip)

    def stop(self):
        if self.writable and self.stop_pub:
            self.stop_pub.publish(self.Empty())  # Supervisory request, not a certified emergency stop.

    def close(self):
        self.stopped = True
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError('ROS spin thread did not terminate')
        self.node.destroy_node()
