import json
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from ament_index_python.packages import get_package_share_directory


def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx),  np.cos(rx)],
    ])

    Ry = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [0.0,         1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0,         0.0,        1.0],
    ])

    return Rz @ Ry @ Rx


def rb5_pose_array_to_T_mm(data):
    if len(data) < 6:
        raise ValueError(
            f"trigger data must have 6 values: "
            f"[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg], got {len(data)}"
        )

    x, y, z, rx, ry, rz = map(float, data[:6])

    T = np.eye(4)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def object_json_to_cam_T_obj_mm(obj):
    pos = obj["position"]
    ori = obj["orientation"]

    p_cam_obj_mm = np.array([
        pos["x"],
        pos["y"],
        pos["z"],
    ]) * 1000.0

    R_cam_obj = np.column_stack([
        np.array(ori["axis_x"], dtype=np.float64),
        np.array(ori["axis_y"], dtype=np.float64),
        np.array(ori["axis_z"], dtype=np.float64),
    ])

    u, _, vt = np.linalg.svd(R_cam_obj)
    R_cam_obj = u @ vt

    if np.linalg.det(R_cam_obj) < 0:
        R_cam_obj[:, 2] *= -1.0

    T = np.eye(4)
    T[:3, :3] = R_cam_obj
    T[:3, 3] = p_cam_obj_mm
    return T


def yaw_deg_from_R_base_obj(R):
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)
    return (yaw_deg + 180.0) % 360.0 - 180.0


class ObjectPoseTransformNode(Node):
    def __init__(self):
        super().__init__("object_pose_transform_node")

        self.declare_parameter("handeye_result_path", "")
        self.declare_parameter("min_confidence", 0.3)

        self.declare_parameter("object_topic", "/object_poses")
        self.declare_parameter("insert_topic", "/insert_poses")
        self.declare_parameter("detect_mode_topic", "/detect_mode")

        self.declare_parameter("peg_trigger_topic", "/manipulation/trigger_peg")
        self.declare_parameter("hole_trigger_topic", "/manipulation/trigger_hole")

        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")

        self.declare_parameter("insert_duplicate_dist_mm", 12.0)
        self.declare_parameter("collect_frames", 5)
        self.declare_parameter("detect_mode_settle_sec", 0.5)

        self.class_to_id = {
            "cylinder": 0,
            "cylinder_insert": 0,
            "hole": 1,
            "hole_insert": 1,
            "cross": 2,
            "cross_insert": 2,
        }

        self.min_confidence = float(self.get_parameter("min_confidence").value)

        self.object_topic = self.get_parameter("object_topic").value
        self.insert_topic = self.get_parameter("insert_topic").value
        self.detect_mode_topic = self.get_parameter("detect_mode_topic").value

        self.peg_output_topic = self.get_parameter("peg_output_topic").value
        self.hole_output_topic = self.get_parameter("hole_output_topic").value

        self.ee_T_cam = self.load_handeye_result_as_mm()

        self.latest_objects = []
        self.latest_inserts = []

        self.pending_task = None
        self.pending_trigger_msg = None

        self.collect_count = 0
        self.collected_targets = []
        self.mode_switch_time = None

        self.detect_mode_pub = self.create_publisher(String, self.detect_mode_topic, 10)

        self.object_sub = self.create_subscription(
            String, self.object_topic, self.object_callback, 10
        )

        self.insert_sub = self.create_subscription(
            String, self.insert_topic, self.insert_callback, 10
        )

        self.peg_trigger_sub = self.create_subscription(
            Float64MultiArray,
            self.get_parameter("peg_trigger_topic").value,
            self.peg_trigger_callback,
            10,
        )

        self.hole_trigger_sub = self.create_subscription(
            Float64MultiArray,
            self.get_parameter("hole_trigger_topic").value,
            self.hole_trigger_callback,
            10,
        )

        self.peg_pub = self.create_publisher(Float64MultiArray, self.peg_output_topic, 10)
        self.hole_pub = self.create_publisher(Float64MultiArray, self.hole_output_topic, 10)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def is_settle_done(self):
        if self.mode_switch_time is None:
            return True

        settle_sec = float(self.get_parameter("detect_mode_settle_sec").value)
        return (self.now_sec() - self.mode_switch_time) >= settle_sec

    def start_collect(self):
        self.collect_count = 0
        self.collected_targets = []
        self.mode_switch_time = self.now_sec()

    def publish_detect_mode(self, mode):
        msg = String()
        msg.data = mode
        self.detect_mode_pub.publish(msg)
        self.get_logger().info(f"request detect_mode: {mode}")

    def load_handeye_result_as_mm(self):
        param_path = self.get_parameter("handeye_result_path").value

        if param_path:
            result_path = Path(param_path)
        else:
            result_path = (
                Path(get_package_share_directory("calib"))
                / "config"
                / "handeye_capture_rs"
                / "handeye_result.json"
            )

        if not result_path.exists():
            raise FileNotFoundError(f"handeye_result.json not found: {result_path}")

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ee_T_cam = np.array(data["ee_T_cam"], dtype=np.float64)
        ee_T_cam_mm = ee_T_cam.copy()
        ee_T_cam_mm[:3, 3] *= 1000.0
        return ee_T_cam_mm

    def object_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_objects = data.get("objects", [])
        except Exception as e:
            self.latest_objects = []
            self.get_logger().warn(f"failed to parse /object_poses JSON: {e}")
            return

        if self.pending_task == "peg_wait_object":
            self.collect_peg_object_frame()

    def insert_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_inserts = data.get("objects", [])
        except Exception as e:
            self.latest_inserts = []
            self.get_logger().warn(f"failed to parse /insert_poses JSON: {e}")
            return

        if self.pending_task == "hole_wait_insert":
            self.collect_hole_insert_frame()

    def transform_one_object_to_xyyaw(self, obj, base_T_ee):
        cam_T_obj = object_json_to_cam_T_obj_mm(obj)
        base_T_obj = base_T_ee @ self.ee_T_cam @ cam_T_obj

        p = base_T_obj[:3, 3]
        R = base_T_obj[:3, :3]

        x_mm = float(p[0])
        y_mm = float(p[1])

        if "yaw_deg" in obj:
            yaw_deg = float(obj["yaw_deg"])
            yaw_source = obj.get("yaw_source", "template")
        else:
            yaw_deg = yaw_deg_from_R_base_obj(R)
            yaw_source = "pca"

        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
        return x_mm, y_mm, float(yaw_deg), yaw_source

    def object_to_target_dict(self, obj, base_T_ee):
        cls = obj.get("class", "")
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            return None

        if cls not in self.class_to_id:
            return None

        x_mm, y_mm, yaw_deg, yaw_source = self.transform_one_object_to_xyyaw(
            obj, base_T_ee
        )

        return {
            "class": cls,
            "id": int(self.class_to_id[cls]),
            "x": float(x_mm),
            "y": float(y_mm),
            "yaw": float(yaw_deg),
            "yaw_source": yaw_source,
            "confidence": float(conf),
        }

    def make_targets_from_objects(self, objects, base_T_ee):
        targets = []
        for obj in objects:
            target = self.object_to_target_dict(obj, base_T_ee)
            if target is not None:
                targets.append(target)
        return targets

    def suppress_duplicate_targets_by_conf(self, targets, dist_thresh_mm):
        kept = []
        targets_sorted = sorted(
            targets,
            key=lambda t: float(t["confidence"]),
            reverse=True,
        )

        for t in targets_sorted:
            duplicate = False

            for k in kept:
                if int(t["id"]) != int(k["id"]):
                    continue

                dist_mm = float(np.hypot(t["x"] - k["x"], t["y"] - k["y"]))

                if dist_mm < dist_thresh_mm:
                    duplicate = True
                    self.get_logger().info(
                        f"suppress duplicate target | "
                        f"drop={t['class']} conf={t['confidence']:.2f} "
                        f"keep={k['class']} conf={k['confidence']:.2f} "
                        f"id={t['id']} dist={dist_mm:.1f}mm"
                    )
                    break

            if not duplicate:
                kept.append(t)

        return kept

    def targets_to_msg_data(self, targets):
        data = []
        for t in targets:
            data.extend([
                float(t["x"]),
                float(t["y"]),
                float(t["yaw"]),
                float(t["id"]),
            ])
        return data

    def publish_repeated(self, publisher, msg, count=10):
        for _ in range(count):
            publisher.publish(msg)

    def reset_pending(self):
        self.pending_task = None
        self.pending_trigger_msg = None
        self.collect_count = 0
        self.collected_targets = []
        self.mode_switch_time = None

    def peg_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(
                f"ignore peg trigger: pending_task={self.pending_task}"
            )
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_task = "peg_wait_object"

        self.start_collect()
        self.publish_detect_mode("object")

    def collect_peg_object_frame(self):
        if not self.is_settle_done():
            return

        base_T_ee = rb5_pose_array_to_T_mm(self.pending_trigger_msg.data)

        targets = self.make_targets_from_objects(self.latest_objects, base_T_ee)
        self.collected_targets.extend(targets)
        self.collect_count += 1

        collect_frames = int(self.get_parameter("collect_frames").value)
        self.get_logger().info(
            f"[COLLECT PEG-OBJECT] frame={self.collect_count}/{collect_frames}, "
            f"targets={len(targets)}, total={len(self.collected_targets)}"
        )

        if self.collect_count < collect_frames:
            return

        final_targets = self.suppress_duplicate_targets_by_conf(
            self.collected_targets,
            dist_thresh_mm=float(self.get_parameter("insert_duplicate_dist_mm").value),
        )

        if not final_targets:
            self.get_logger().warn(
                "peg trigger: no valid object target after collection"
            )
            self.reset_pending()
            return

        msg = Float64MultiArray()
        msg.data = self.targets_to_msg_data(final_targets)

        self.get_logger().info(
            f"[PUBLISH] topic={self.peg_output_topic} "
            f"type=object(collected) targets={final_targets} data={msg.data}"
        )

        self.publish_repeated(self.peg_pub, msg, count=10)
        self.reset_pending()

    def hole_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(
                f"ignore hole trigger: pending_task={self.pending_task}"
            )
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_task = "hole_wait_insert"

        # hole trigger는 object를 보지 않고 바로 insert만 수집한다.
        # detect_mode 변경 후 settle time은 기존 start_collect/is_settle_done 로직 그대로 적용된다.
        self.start_collect()
        self.publish_detect_mode("insert")

    def collect_hole_insert_frame(self):
        if not self.is_settle_done():
            return

        base_T_ee = rb5_pose_array_to_T_mm(self.pending_trigger_msg.data)

        targets = self.make_targets_from_objects(self.latest_inserts, base_T_ee)
        self.collected_targets.extend(targets)
        self.collect_count += 1

        collect_frames = int(self.get_parameter("collect_frames").value)
        self.get_logger().info(
            f"[COLLECT HOLE-INSERT] frame={self.collect_count}/{collect_frames}, "
            f"targets={len(targets)}, total={len(self.collected_targets)}"
        )

        if self.collect_count < collect_frames:
            return

        final_targets = self.suppress_duplicate_targets_by_conf(
            self.collected_targets,
            dist_thresh_mm=float(self.get_parameter("insert_duplicate_dist_mm").value),
        )

        if not final_targets:
            self.get_logger().warn(
                "hole trigger: no valid insert target after collection"
            )
            self.reset_pending()
            return

        msg = Float64MultiArray()
        msg.data = self.targets_to_msg_data(final_targets)

        self.get_logger().info(
            f"[PUBLISH] topic={self.hole_output_topic} "
            f"type=insert(collected) targets={final_targets} data={msg.data}"
        )

        self.publish_repeated(self.hole_pub, msg, count=10)
        self.reset_pending()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPoseTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()