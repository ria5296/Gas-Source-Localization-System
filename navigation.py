#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, Float32MultiArray
import rosbag2_py
import transforms3d.euler as t3e
import math
from multiprocessing import shared_memory
import struct

BAG_PATH = "/home/robotics/my_amcl_bag"  # rosbag2 경로 (amcl_pose 포함)

class BagNav2Follower(Node):
    def __init__(self):
        super().__init__("bag_nav2_follower")

        # Action client (Nav2)
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Publishers
        self.done_pub = self.create_publisher(Bool, "/patrol_done", 10)
        self.status_pub = self.create_publisher(Float32MultiArray, "/robot_status", 10)

        # State
        self._published_done = False
        self.current_pose = None
        self.current_gas = 0.0

        # Load waypoints from bag
        self.point_array = self.load_bag_waypoints(BAG_PATH)
        self._waypoint_idx = 0
        self._navigating = False
        self.get_logger().info(f"✅ Loaded {len(self.point_array)} waypoints from bag")

        # 시작 시 False 발행
        self.publish_done(False)

        # Timers
        self.timer = self.create_timer(1.0, self.main_loop)        # goal 전송 루프 (1Hz)
        self.status_timer = self.create_timer(3.0, self.publish_status)  # 상태 발행 루프 (3초)

        # Subscribers
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.amcl_callback, 10)

        # Shared memory (gas sensor)
        try:
            self.gas_shm = shared_memory.SharedMemory(name="gas_sensor")
        except FileNotFoundError:
            self.get_logger().warn("⚠️ Shared memory 'gas_sensor' not found. Gas values will be N/A.")
            self.gas_shm = None

    # ---------- Helpers ----------
    def publish_done(self, value: bool):
        msg = Bool()
        msg.data = value
        self.done_pub.publish(msg)
        self.get_logger().info(f"📢 /patrol_done = {value}")

    def publish_status(self):
        # 3초마다 [x, y, gas] 발행
        if self.current_pose is None:
            return

        x = self.current_pose.pose.pose.position.x
        y = self.current_pose.pose.pose.position.y

        gas_val = self.current_gas
        if self.gas_shm:
            try:
                gas_bytes = self.gas_shm.buf[:4]
                gas_val = struct.unpack('f', gas_bytes)[0]
                self.current_gas = gas_val
            except Exception as e:
                self.get_logger().warn(f"Gas read error: {e}")

        msg = Float32MultiArray()
        msg.data = [float(x), float(y), float(gas_val)]
        self.status_pub.publish(msg)

    # ---------- Callbacks ----------
    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg  # 최신 pose 저장

        gas_val = None
        if self.gas_shm:
            try:
                gas_bytes = self.gas_shm.buf[:4]
                gas_val = struct.unpack('f', gas_bytes)[0]
                self.current_gas = gas_val
            except Exception:
                gas_val = None

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        if gas_val is not None:
            print(f"AMCL Pose → x={pos.x:.3f}, y={pos.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f} | Gas: {gas_val:.1f}")
        else:
            print(f"AMCL Pose → x={pos.x:.3f}, y={pos.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f} | Gas: N/A")

    # ---------- Waypoints from rosbag2 ----------
    def load_bag_waypoints(self, bag_path):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {t.name: t.type for t in topic_types}
        if "/amcl_pose" not in topic_type_map:
            self.get_logger().error("❌ /amcl_pose topic not found in bag")
            return []

        import rclpy.serialization as serial
        raw_waypoints = []

        while reader.has_next():
            topic, data, _t = reader.read_next()
            if topic != "/amcl_pose":
                continue
            msg = serial.deserialize_message(data, PoseWithCovarianceStamped)
            q = msg.pose.pose.orientation
            _, _, yaw = t3e.quat2euler([q.w, q.x, q.y, q.z])
            raw_waypoints.append((
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0)
            ))

        if not raw_waypoints:
            return []

        # 20cm 이상 이동한 지점만 waypoint로 샘플링
        waypoints = [raw_waypoints[0]]
        for pt in raw_waypoints[1:]:
            last = waypoints[-1]
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) >= 0.2:
                waypoints.append(pt)

        return waypoints

    # ---------- Nav2 Goal Send ----------
    def send_goal(self, pose_tuple):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("❌ navigate_to_pose action server not available")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = pose_tuple[0]
        goal_msg.pose.pose.position.y = pose_tuple[1]
        goal_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=pose_tuple[2], w=pose_tuple[3])

        self.get_logger().info(f"📍 Sending goal {self._waypoint_idx}: x={pose_tuple[0]:.2f}, y={pose_tuple[1]:.2f}")
        self._navigating = True
        future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("❌ Goal rejected")
            self._navigating = False
            return

        self.get_logger().info("✅ Goal accepted, waiting for result...")
        future = goal_handle.get_result_async()
        future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # 필요시 feedback 처리
        pass

    def get_result_callback(self, future):
        self.get_logger().info(f"🏁 Waypoint {self._waypoint_idx} reached")
        self._waypoint_idx += 1
        self._navigating = False

        # 모든 waypoint 완료 시 True 발행(한 번만)
        if self._waypoint_idx >= len(self.point_array) and not self._published_done:
            self.get_logger().info("🎉 All waypoints completed — publishing True to /patrol_done")
            self.publish_done(True)
            self._published_done = True

    def main_loop(self):
        if not self.point_array:
            return
        if self._navigating or self._waypoint_idx >= len(self.point_array):
            return
        self.send_goal(self.point_array[self._waypoint_idx])

def main():
    rclpy.init()
    node = BagNav2Follower()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()