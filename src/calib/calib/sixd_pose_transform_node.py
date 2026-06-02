"""
object / peg output:
Success, len(data) == 17:
[
  target_T_00, target_T_01, target_T_02, target_T_03,
  target_T_10, target_T_11, target_T_12, target_T_13,
  target_T_20, target_T_21, target_T_22, target_T_23,
  target_T_30, target_T_31, target_T_32, target_T_33,
  requested_object_id
]

Failure / requested object pose not available, len(data) != 17:
[
  currently_visible_object_id_0,
  currently_visible_object_id_1,
  ...
]

- target_T is base_T_object by default.
- Later, set peg_target_pose_mode := "grasp" and provide object_grasp_yaml_path
  to publish base_T_grasp instead.

insert / hole output:
[
  x_mm,
  y_mm,
  yaw_deg,
  id
]

peg trigger input:
[
  object_id,
  tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg
]

hole trigger input remains unchanged:
[
  tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg
]
"""



import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64MultiArray
from ament_index_python.packages import get_package_share_directory


# ============================================================
# Rotation / Transform util

#  rot3 to R
def euler_zyx_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rx), -np.sin(rx)],
        [0.0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0.0, np.sin(ry)],
        [0.0,         1.0, 0.0],
        [-np.sin(ry), 0.0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz),  np.cos(rz), 0.0],
        [0.0,         0.0,        1.0],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

# R to rot3
def R_to_euler_zyx_deg(R: np.ndarray) -> np.ndarray:

    R = orthonormalize_R(np.asarray(R, dtype=np.float64).reshape(3, 3))

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy >= 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    return np.degrees([rx, ry, rz])

# Q to R
def quat_xyzw_to_R(q_xyzw: List[float]) -> np.ndarray:

    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q

    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid quaternion: norm is zero.")
    x, y, z, w = q / n

    R = np.array([
        [1.0 - 2.0 * (y*y + z*z),       2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
        [      2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z),       2.0 * (y*z - x*w)],
        [      2.0 * (x*z - y*w),       2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)

    return orthonormalize_R(R)


def orthonormalize_R(R: np.ndarray) -> np.ndarray:
    """
    Project a nearly-rotation matrix to SO(3).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt


def validate_T(T: np.ndarray, name: str = "T", atol: float = 1e-3):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)

    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name}: last row must be [0, 0, 0, 1]. got {T[3]}")

    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(f"{name}: det(R) must be close to 1. det={det}")

    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name}: R.T @ R must be close to I.")


def pose6_mm_deg_to_T_mm(pose6) -> np.ndarray:
    """
    [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] -> 4x4 transform in mm.
    """
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.size < 6:
        raise ValueError(
            f"pose6 must have at least 6 values: "
            f"[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg], got {pose6.size}"
        )

    x, y, z, rx, ry, rz = pose6[:6]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def T_mm_to_pose6_mm_deg(T: np.ndarray) -> np.ndarray:
    """
    4x4 transform in mm -> [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    validate_T(T, name="T_mm_to_pose6 input")

    R = orthonormalize_R(T[:3, :3])
    rpy = R_to_euler_zyx_deg(R)
    p = T[:3, 3]

    return np.array([p[0], p[1], p[2], rpy[0], rpy[1], rpy[2]], dtype=np.float64)


def T_m_to_T_mm(T_m: np.ndarray) -> np.ndarray:
    T = np.asarray(T_m, dtype=np.float64).reshape(4, 4).copy()
    T[:3, 3] *= 1000.0
    return T


def T_mm_to_T_m(T_mm: np.ndarray) -> np.ndarray:
    T = np.asarray(T_mm, dtype=np.float64).reshape(4, 4).copy()
    T[:3, 3] /= 1000.0
    return T


def matrix_from_data(data, unit: str = "mm") -> np.ndarray:
    """
    4x4 matrix with translation unit either mm or m.
    Return transform in mm.
    """
    T = np.asarray(data, dtype=np.float64).reshape(4, 4).copy()
    T[:3, :3] = orthonormalize_R(T[:3, :3])

    if unit == "m":
        T[:3, 3] *= 1000.0
    elif unit == "mm":
        pass
    else:
        raise ValueError(f"Unsupported unit: {unit}. Use 'm' or 'mm'.")

    validate_T(T, name="matrix_from_data")
    return T


def normalize_vec(v: np.ndarray, name: str = "vector") -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"{name}: norm is too small.")
    return v / n


def canonicalize_object_axes_z_up_xy_order(
    T_base_obj: np.ndarray,
    up_axis_base: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64),
    z_flip_margin: float = 0.05,
    xy_down_margin: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Canonicalize object frame axes with respect to the base/world ground plane.

    Assumption:
        base/world +Z is upward, so a negative dot with +Z means
        the axis points into the ground.

    Rules:
      1) If object +Z points into the ground, use the opposite direction as +Z.
         To preserve a right-handed frame, +X is flipped together with +Z.

      2) Between object +X and +Y, if +X points more into the ground than +Y,
         swap the planar axis role while preserving right-handedness:
             new_x = old_y
             new_y = -old_x
             new_z = old_z

    Translation is not changed.
    """
    T = np.asarray(T_base_obj, dtype=np.float64).reshape(4, 4).copy()
    validate_T(T, name="canonicalize input")

    up = normalize_vec(up_axis_base, name="up_axis_base")

    R = orthonormalize_R(T[:3, :3])
    x = R[:, 0].copy()
    y = R[:, 1].copy()
    z = R[:, 2].copy()

    info: Dict[str, Any] = {
        "z_flipped": False,
        "xy_swapped": False,
        "before_dot_up": {
            "x": float(np.dot(x, up)),
            "y": float(np.dot(y, up)),
            "z": float(np.dot(z, up)),
        },
    }

    z_dot_up = float(np.dot(z, up))

    # Rule 1: object +Z must not point into the ground.
    # Use a small margin so near-horizontal/noisy axes do not flip randomly.
    if z_dot_up < -float(z_flip_margin):
        # [x, y, z] -> [-x, y, -z] keeps det(R)=+1.
        x = -x
        z = -z
        info["z_flipped"] = True

    # Rule 2: Compare only the downward component of +X and +Y.
    # down_score = 0 means not pointing into the ground.
    # Larger down_score means closer to base/world -Z.
    down_x = max(0.0, -float(np.dot(x, up)))
    down_y = max(0.0, -float(np.dot(y, up)))
    info["down_score"] = {"x": float(down_x), "y": float(down_y)}

    if down_x > down_y + float(xy_down_margin):
        old_x = x.copy()
        old_y = y.copy()
        x = old_y
        y = -old_x
        info["xy_swapped"] = True

    # Rebuild a clean right-handed orthonormal frame while preserving chosen +Z and +X.
    z = normalize_vec(z, name="canonicalized z")
    x = x - z * float(np.dot(x, z))
    x = normalize_vec(x, name="canonicalized x")
    y = np.cross(z, x)
    y = normalize_vec(y, name="canonicalized y")

    R_new = np.column_stack([x, y, z])
    R_new = orthonormalize_R(R_new)

    T[:3, :3] = R_new
    validate_T(T, name="canonicalized base_T_obj")

    info["after_dot_up"] = {
        "x": float(np.dot(T[:3, 0], up)),
        "y": float(np.dot(T[:3, 1], up)),
        "z": float(np.dot(T[:3, 2], up)),
    }
    return T, info


def object_json_to_cam_T_obj_mm(obj: Dict[str, Any]) -> np.ndarray:
    """
    Convert object pose JSON from FoundationPose / detector node to camera_T_object in mm.

    Supported formats:
    1) Preferred:
        obj["pose_matrix"] = 4x4 camera_T_object, translation in meters.

    2) Position + quaternion:
        obj["position"] = {"x": m, "y": m, "z": m}
        obj["orientation"] = {"x": qx, "y": qy, "z": qz, "w": qw}

    3) Legacy axis format:
        obj["position"] = {"x": m, "y": m, "z": m}
        obj["orientation"] = {
            "axis_x": [...],
            "axis_y": [...],
            "axis_z": [...]
        }
    """
    if "pose_matrix" in obj:
        T_cam_obj_m = np.asarray(obj["pose_matrix"], dtype=np.float64).reshape(4, 4)
        T_cam_obj_m[:3, :3] = orthonormalize_R(T_cam_obj_m[:3, :3])
        validate_T(T_cam_obj_m, name="obj.pose_matrix")
        return T_m_to_T_mm(T_cam_obj_m)

    if "position" not in obj or "orientation" not in obj:
        raise KeyError("object must contain either 'pose_matrix' or both 'position' and 'orientation'.")

    pos = obj["position"]
    ori = obj["orientation"]

    p_cam_obj_mm = np.array([
        float(pos["x"]),
        float(pos["y"]),
        float(pos["z"]),
    ], dtype=np.float64) * 1000.0

    if all(k in ori for k in ("x", "y", "z", "w")):
        R_cam_obj = quat_xyzw_to_R([
            float(ori["x"]),
            float(ori["y"]),
            float(ori["z"]),
            float(ori["w"]),
        ])
    elif all(k in ori for k in ("axis_x", "axis_y", "axis_z")):
        R_cam_obj = np.column_stack([
            np.asarray(ori["axis_x"], dtype=np.float64).reshape(3),
            np.asarray(ori["axis_y"], dtype=np.float64).reshape(3),
            np.asarray(ori["axis_z"], dtype=np.float64).reshape(3),
        ])
        R_cam_obj = orthonormalize_R(R_cam_obj)
    else:
        raise KeyError(
            "orientation must be quaternion keys x/y/z/w or axis_x/axis_y/axis_z."
        )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam_obj
    T[:3, 3] = p_cam_obj_mm

    validate_T(T, name="cam_T_obj")
    return T



def canonical_object_name(cls: str) -> str:
    """
    Insert class shares object CAD/grasp config by default.
    """
    alias = {
        "cross_insert": "cross",
        "cylinder_insert": "cylinder",
        "hole_insert": "hole",
    }
    return alias.get(cls, cls)



# ============================================================
# ROS2 Node
# ============================================================

class ObjectPoseTransformNode(Node):
    """
    Mixed output transform node.

    Input:
        /manipulation/trigger_peg:
            std_msgs/Float64MultiArray
            data = [object_id, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            object_id is mapped to class name and relayed to the 6D pose trigger topic.

        /object_poses:
            Object detections from mixed_pose_vision_node.
            Object mode is expected to use FoundationPose pose_matrix.

        /manipulation/trigger_hole:
            std_msgs/Float64MultiArray
            data = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            This keeps the old hole/insert behavior.

        /insert_poses:
            Insert detections from mixed_pose_vision_node.
            Insert mode keeps the old output convention: [x, y, yaw, id].

    Output:
        /vision/peg_targets:
            Success: 17 floats:
                [base_T_target row-major 16 values, requested_object_id]

            Failure / requested object pose not available: variable-length id list:
                [currently_visible_object_id_0, currently_visible_object_id_1, ...]

            Therefore the controller can branch by len(data):
                len(data) == 17 -> success matrix response
                len(data) != 17 -> failure/fallback visible-id response

            base_T_target is base_T_object by default.
            For future grasp output, set peg_target_pose_mode="grasp"
            and provide object_grasp_yaml_path with object_to_grasp transforms.

        /vision/hole_targets:
            insert target, legacy format. 4 floats per target:
            [x_mm, y_mm, yaw_deg, id]
    """

    def __init__(self):
        super().__init__("sixd_pose_transform_node")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("handeye_result_path", "")

        # Optional future extension:
        # - peg_target_pose_mode="object" publishes base_T_object.
        # - peg_target_pose_mode="grasp" publishes base_T_object @ object_T_grasp.
        self.declare_parameter("object_grasp_yaml_path", "")
        self.declare_parameter("peg_target_pose_mode", "object")  # object | grasp

        # Axis canonicalization is applied to the final target frame, not to the raw
        # object frame before object_T_grasp. This preserves the meaning of
        # object_T_grasp as a transform defined in the raw/CAD object frame.
        #
        # object mode:
        #     base_T_obj_raw -> optional canonicalized base_T_object
        # grasp mode:
        #     base_T_obj_raw @ object_T_grasp -> optional canonicalized base_T_grasp
        self.declare_parameter("canonicalize_object_axes", True)
        self.declare_parameter("canonicalize_grasp_axes", True)
        self.declare_parameter("canonicalize_z_flip_margin", 0.05)
        self.declare_parameter("canonicalize_xy_down_margin", 0.05)

        self.declare_parameter("min_confidence", 0.3)
        self.declare_parameter("object_topic", "/object_poses")
        self.declare_parameter("insert_topic", "/insert_poses")
        self.declare_parameter("detect_mode_topic", "/detect_mode")

        self.declare_parameter("peg_trigger_topic", "/manipulation/trigger_peg")
        self.declare_parameter("hole_trigger_topic", "/manipulation/trigger_hole")

        # Topic used to request a class-specific 6D pose estimation.
        # The message is String.data = "cylinder" | "hole" | "cross" by default.
        self.declare_parameter("object_6d_trigger_topic", "/manipulation/object_6d_trigger")

        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")


        self.class_to_id = {
            "cylinder": 0,
            "cylinder_insert": 0,
            "hole": 1,
            "hole_insert": 1,
            "cross": 2,
            "cross_insert": 2,
        }
        self.id_to_class = {
            0: "cylinder",
            1: "hole",
            2: "cross",
        }

        self.min_confidence = float(self.get_parameter("min_confidence").value)

        self.object_topic = str(self.get_parameter("object_topic").value)
        self.insert_topic = str(self.get_parameter("insert_topic").value)
        self.detect_mode_topic = str(self.get_parameter("detect_mode_topic").value)
        self.object_6d_trigger_topic = str(self.get_parameter("object_6d_trigger_topic").value)
        self.peg_output_topic = str(self.get_parameter("peg_output_topic").value)
        self.hole_output_topic = str(self.get_parameter("hole_output_topic").value)

        # ----------------------------
        # Load transforms/config
        # ----------------------------
        self.ee_T_cam = self.load_handeye_result_as_mm()
        self.grasp_cfg = self.load_object_grasp_config()

        # ----------------------------
        # Runtime state
        # ----------------------------
        self.latest_objects = []
        self.latest_object_available_ids = []
        self.latest_object_status = ""
        self.latest_inserts = []

        self.pending_task = None
        self.pending_trigger_msg = None
        self.pending_object_idx = None

        # ----------------------------
        # ROS I/O
        # ----------------------------
        self.detect_mode_pub = self.create_publisher(String, self.detect_mode_topic, 10)
        self.object_6d_trigger_pub = self.create_publisher(String, self.object_6d_trigger_topic, 10)

        self.object_sub = self.create_subscription(
            String, self.object_topic, self.object_callback, 10
        )
        self.insert_sub = self.create_subscription(
            String, self.insert_topic, self.insert_callback, 10
        )

        self.peg_trigger_sub = self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("peg_trigger_topic").value),
            self.peg_trigger_callback,
            10,
        )
        self.hole_trigger_sub = self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("hole_trigger_topic").value),
            self.hole_trigger_callback,
            10,
        )

        self.peg_pub = self.create_publisher(Float64MultiArray, self.peg_output_topic, 10)
        self.hole_pub = self.create_publisher(Float64MultiArray, self.hole_output_topic, 10)

        self.get_logger().info(
            "ObjectPoseTransformNode ready. "
            "peg trigger=/manipulation/trigger_peg: [object_id, tcp_pose6]; "
            "object output=/vision/peg_targets: success len=17 [target_T_row_major_16, id], failure len!=17 [visible_ids]; "
            "insert output=/vision/hole_targets: [x_mm, y_mm, yaw_deg, id]."
        )

    # ============================================================
    # State helpers
    # ============================================================

    def reset_pending(self):
        self.pending_task = None
        self.pending_trigger_msg = None
        self.pending_object_idx = None

    def publish_detect_mode(self, mode):
        msg = String()
        msg.data = mode
        self.detect_mode_pub.publish(msg)
        self.get_logger().info(f"request detect_mode: {mode}")

    def publish_object_6d_trigger(self, object_idx: int):
        class_name = self.class_name_from_id(object_idx)
        msg = String()
        msg.data = class_name
        self.object_6d_trigger_pub.publish(msg)
        self.get_logger().info(
            f"request object 6D pose: topic={self.object_6d_trigger_topic}, "
            f"id={object_idx}, class={class_name}"
        )
        return class_name

    def class_name_from_id(self, object_idx: int) -> str:
        if int(object_idx) not in self.id_to_class:
            raise ValueError(
                f"unknown object_id={object_idx}. "
                f"valid ids={sorted(self.id_to_class.keys())}"
            )
        return self.id_to_class[int(object_idx)]

    # ============================================================
    # Config loading
    # ============================================================

    def load_handeye_result_as_mm(self) -> np.ndarray:
        """
        Load ee_T_cam from handeye_result.json.
        Assumption:
            handeye_result.json stores translation in meters.
        Return:
            ee_T_cam in mm.
        """
        param_path = str(self.get_parameter("handeye_result_path").value)

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

        if "ee_T_cam" not in data:
            raise KeyError(f"'ee_T_cam' not found in {result_path}")

        ee_T_cam_m = np.asarray(data["ee_T_cam"], dtype=np.float64).reshape(4, 4)
        ee_T_cam_m[:3, :3] = orthonormalize_R(ee_T_cam_m[:3, :3])
        validate_T(ee_T_cam_m, name="ee_T_cam_m")

        ee_T_cam_mm = T_m_to_T_mm(ee_T_cam_m)
        validate_T(ee_T_cam_mm, name="ee_T_cam_mm")

        self.get_logger().info(f"loaded hand-eye: {result_path}")
        return ee_T_cam_mm

    def load_object_grasp_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Optional YAML format for future grasp output:

        unit: mm
        objects:
          cylinder:
            object_to_grasp:
              unit: mm
              matrix:
                - [1, 0, 0, 0]
                - [0, 1, 0, 0]
                - [0, 0, 1, 0]
                - [0, 0, 0, 1]

        This node currently publishes object pose by default.
        To publish grasp pose later, set peg_target_pose_mode="grasp".
        """
        yaml_path_str = str(self.get_parameter("object_grasp_yaml_path").value)
        cfg: Dict[str, Dict[str, Any]] = {}

        if not yaml_path_str:
            self.get_logger().warn(
                "object_grasp_yaml_path is empty. "
                "Using identity object_to_grasp if peg_target_pose_mode='grasp'."
            )
            return cfg

        path = Path(yaml_path_str)
        if not path.exists():
            raise FileNotFoundError(f"object_grasp_yaml_path not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        global_unit = data.get("unit", "mm")
        objects = data.get("objects", {})
        if not isinstance(objects, dict):
            raise ValueError("object_grasp_yaml: 'objects' must be a dict.")

        for name, obj_data in objects.items():
            if obj_data is None:
                obj_data = {}

            grasp_data = obj_data.get("object_to_grasp", {})
            unit = grasp_data.get("unit", obj_data.get("unit", global_unit))

            if "matrix" in grasp_data:
                T_obj_grasp = matrix_from_data(grasp_data["matrix"], unit=unit)
            elif "matrix" in obj_data:
                T_obj_grasp = matrix_from_data(obj_data["matrix"], unit=unit)
            else:
                T_obj_grasp = np.eye(4, dtype=np.float64)

            cfg[str(name)] = {
                "object_to_grasp": T_obj_grasp,
            }

        self.get_logger().info(f"loaded object grasp YAML: {path}")
        return cfg

    def get_object_to_grasp_for_class(self, cls: str) -> np.ndarray:
        base_name = canonical_object_name(cls)

        if cls in self.grasp_cfg:
            item = self.grasp_cfg[cls]
        elif base_name in self.grasp_cfg:
            item = self.grasp_cfg[base_name]
        else:
            item = {
                "object_to_grasp": np.eye(4, dtype=np.float64),
            }

        T_obj_grasp = np.asarray(item["object_to_grasp"], dtype=np.float64).reshape(4, 4)
        validate_T(T_obj_grasp, name=f"object_T_grasp[{cls}]")
        return T_obj_grasp.copy()

    # ============================================================
    # Input callbacks
    # ============================================================

    def object_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_objects = self.extract_objects_from_payload(data)
            self.latest_object_available_ids = self.extract_available_ids_from_payload(data)
            self.latest_object_status = str(data.get("status", ""))
        except Exception as e:
            self.latest_objects = []
            self.latest_object_available_ids = []
            self.latest_object_status = "parse_error"
            self.get_logger().warn(f"failed to parse {self.object_topic} JSON: {e}")
            return

        if self.pending_task == "peg_wait_object":
            self.collect_peg_object_frame()

    def insert_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_inserts = self.extract_objects_from_payload(data)
        except Exception as e:
            self.latest_inserts = []
            self.get_logger().warn(f"failed to parse {self.insert_topic} JSON: {e}")
            return

        if self.pending_task == "hole_wait_insert":
            self.collect_hole_insert_frame()

    @staticmethod
    def extract_objects_from_payload(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Supports perception JSON:
            {"objects": [...]}
        or:
            {"target": {...}, "objects": [...]}
        """
        objects = data.get("objects", [])
        if objects is None:
            objects = []

        if not isinstance(objects, list):
            raise ValueError("'objects' must be a list.")

        # If only target exists, use it.
        if len(objects) == 0 and isinstance(data.get("target"), dict):
            objects = [data["target"]]

        return objects

    def extract_available_ids_from_payload(self, data: Dict[str, Any]) -> List[int]:
        """
        Return currently visible object ids from perception JSON.

        Preferred source:
            data["available_ids"] from mixed_pose_vision_node.

        Fallback source:
            class names inside data["objects"].
        """
        raw_ids = data.get("available_ids", None)
        ids = []

        if isinstance(raw_ids, list):
            for v in raw_ids:
                try:
                    obj_id = int(round(float(v)))
                except Exception:
                    continue
                if obj_id in self.id_to_class and obj_id not in ids:
                    ids.append(obj_id)

        if ids:
            return ids

        try:
            objects = self.extract_objects_from_payload(data)
        except Exception:
            objects = []

        for obj in objects:
            cls = str(obj.get("class", ""))
            if cls not in self.class_to_id:
                continue
            obj_id = int(self.class_to_id[cls])
            if obj_id not in ids:
                ids.append(obj_id)

        return ids

    # ============================================================
    # Transform core - object output: matrix + id
    # ============================================================

    def object_to_matrix_target_dict(
        self,
        obj: Dict[str, Any],
        base_T_ee: np.ndarray,
        requested_object_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        cls = str(obj.get("class", ""))
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            return None

        if cls not in self.class_to_id:
            return None

        obj_id = int(self.class_to_id[cls])
        if requested_object_idx is not None and obj_id != int(requested_object_idx):
            return None

        try:
            cam_T_obj = object_json_to_cam_T_obj_mm(obj)
            base_T_obj_raw = base_T_ee @ self.ee_T_cam @ cam_T_obj
            validate_T(base_T_obj_raw, name=f"base_T_obj_raw[{cls}]")

            target_mode = str(self.get_parameter("peg_target_pose_mode").value).strip().lower()

            if target_mode == "object":
                # Object output: canonicalization is applied directly to the object frame.
                if bool(self.get_parameter("canonicalize_object_axes").value):
                    base_T_target, axis_info = canonicalize_object_axes_z_up_xy_order(
                        base_T_obj_raw,
                        z_flip_margin=float(self.get_parameter("canonicalize_z_flip_margin").value),
                        xy_down_margin=float(self.get_parameter("canonicalize_xy_down_margin").value),
                    )
                    if axis_info["z_flipped"] or axis_info["xy_swapped"]:
                        self.get_logger().info(
                            f"canonicalized object target axes class={cls} "
                            f"z_flipped={axis_info['z_flipped']} "
                            f"xy_swapped={axis_info['xy_swapped']} "
                            f"before_dot_up={axis_info['before_dot_up']} "
                            f"down_score={axis_info.get('down_score', {})} "
                            f"after_dot_up={axis_info['after_dot_up']}"
                        )
                else:
                    base_T_target = base_T_obj_raw

                validate_T(base_T_target, name=f"base_T_object_target[{cls}]")
                target_frame = "object"

            elif target_mode == "grasp":
                # Grasp output: first apply object_T_grasp in the raw/CAD object frame,
                # then canonicalize the resulting grasp frame in base/world coordinates.
                # This is the intended pipeline:
                #     RAW object -> 1st grasp -> final axis-canonical grasp
                object_T_grasp = self.get_object_to_grasp_for_class(cls)
                base_T_grasp_raw = base_T_obj_raw @ object_T_grasp
                validate_T(base_T_grasp_raw, name=f"base_T_grasp_raw[{cls}]")

                if bool(self.get_parameter("canonicalize_grasp_axes").value):
                    base_T_target, axis_info = canonicalize_object_axes_z_up_xy_order(
                        base_T_grasp_raw,
                        z_flip_margin=float(self.get_parameter("canonicalize_z_flip_margin").value),
                        xy_down_margin=float(self.get_parameter("canonicalize_xy_down_margin").value),
                    )
                    if axis_info["z_flipped"] or axis_info["xy_swapped"]:
                        self.get_logger().info(
                            f"canonicalized grasp target axes class={cls} "
                            f"z_flipped={axis_info['z_flipped']} "
                            f"xy_swapped={axis_info['xy_swapped']} "
                            f"before_dot_up={axis_info['before_dot_up']} "
                            f"down_score={axis_info.get('down_score', {})} "
                            f"after_dot_up={axis_info['after_dot_up']}"
                        )
                else:
                    base_T_target = base_T_grasp_raw

                validate_T(base_T_target, name=f"base_T_grasp_target[{cls}]")
                target_frame = "grasp"

            else:
                raise ValueError("peg_target_pose_mode must be 'object' or 'grasp'.")

            p = base_T_target[:3, 3]
            return {
                "class": cls,
                "id": obj_id,
                "confidence": conf,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "target_T": base_T_target,
                "target_frame": target_frame,
                "output_format": "matrix16_id_17",
            }

        except Exception as e:
            self.get_logger().warn(f"failed to transform object class={cls}: {e}")
            return None

    # ============================================================
    # Transform core - insert output: legacy [x, y, yaw, id]
    # ============================================================

    def insert_to_legacy_target_dict(self, obj: Dict[str, Any], base_T_ee: np.ndarray) -> Optional[Dict[str, Any]]:
        cls = str(obj.get("class", ""))
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            return None

        if cls not in self.class_to_id:
            return None

        try:
            cam_T_obj = object_json_to_cam_T_obj_mm(obj)
            base_T_obj = base_T_ee @ self.ee_T_cam @ cam_T_obj
            validate_T(base_T_obj, name=f"base_T_obj_insert[{cls}]")

            p = base_T_obj[:3, 3]
            x_mm = float(p[0])
            y_mm = float(p[1])

            if "yaw_deg" in obj:
                yaw_deg = float(obj["yaw_deg"])
                yaw_source = str(obj.get("yaw_source", "template"))
            else:
                yaw_deg = self.yaw_deg_from_R_base_obj(base_T_obj[:3, :3])
                yaw_source = "pose_matrix"

            yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

            return {
                "class": cls,
                "id": int(self.class_to_id[cls]),
                "confidence": conf,
                "x": x_mm,
                "y": y_mm,
                "yaw": float(yaw_deg),
                "yaw_source": yaw_source,
                "source_yaw_deg": obj.get("yaw_deg", None),
                "source_yaw_score": obj.get("yaw_score", None),
                "output_format": "legacy_xyyaw_4",
            }

        except Exception as e:
            self.get_logger().warn(f"failed to transform insert class={cls}: {e}")
            return None

    @staticmethod
    def yaw_deg_from_R_base_obj(R: np.ndarray) -> float:
        R = orthonormalize_R(np.asarray(R, dtype=np.float64).reshape(3, 3))
        yaw_rad = np.arctan2(R[1, 0], R[0, 0])
        yaw_deg = np.degrees(yaw_rad)
        return float((yaw_deg + 180.0) % 360.0 - 180.0)

    def make_matrix_targets_from_objects(self, objects, base_T_ee, requested_object_idx=None):
        targets = []
        for obj in objects:
            target = self.object_to_matrix_target_dict(
                obj,
                base_T_ee,
                requested_object_idx=requested_object_idx,
            )
            if target is not None:
                targets.append(target)
        return targets

    def make_legacy_targets_from_inserts(self, objects, base_T_ee):
        targets = []
        for obj in objects:
            target = self.insert_to_legacy_target_dict(obj, base_T_ee)
            if target is not None:
                targets.append(target)
        return targets

    @staticmethod
    def object_targets_to_msg_data(targets):
        """
        Object target format:
            target_T row-major 16 values + id
        Total 17 floats per target.
        """
        data = []

        for t in targets:
            target_T = np.asarray(t["target_T"], dtype=np.float64).reshape(4, 4)
            data.extend(target_T.reshape(-1).tolist())
            data.append(float(t["id"]))

        return data

    @staticmethod
    def object_visible_ids_to_msg_data(visible_ids):
        """
        Peg fallback/failure format:
            [currently_visible_object_id_0, currently_visible_object_id_1, ...]

        The controller distinguishes this from success by length:
            len(data) == 17 -> success matrix response
            len(data) != 17 -> fallback visible-id response
        """
        ids = []
        for v in visible_ids:
            obj_id = int(round(float(v)))
            if obj_id not in ids:
                ids.append(obj_id)
        return [float(v) for v in ids]

    @staticmethod
    def insert_targets_to_msg_data(targets):
        """
        Insert legacy target format:
            x_mm, y_mm, yaw_deg, id
        Total 4 floats per target.
        """
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

    # ============================================================
    # Trigger handling
    # ============================================================

    @staticmethod
    def parse_peg_trigger_data(trigger_msg) -> Tuple[int, np.ndarray]:
        data = np.asarray(trigger_msg.data, dtype=np.float64).reshape(-1)
        if data.size < 7:
            raise ValueError(
                "peg trigger data must be "
                "[object_id, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]. "
                f"got {data.size} values"
            )

        object_idx = int(round(float(data[0])))
        base_T_ee = pose6_mm_deg_to_T_mm(data[1:7])
        validate_T(base_T_ee, name="base_T_ee")
        return object_idx, base_T_ee

    def peg_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(f"ignore peg trigger: pending_task={self.pending_task}")
            return

        try:
            object_idx, _ = self.parse_peg_trigger_data(trigger_msg)
            self.class_name_from_id(object_idx)
        except Exception as e:
            self.get_logger().warn(f"invalid peg trigger: {e}")
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_object_idx = object_idx
        self.pending_task = "peg_wait_object"

        self.publish_detect_mode("object")
        self.publish_object_6d_trigger(object_idx)

    def hole_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(f"ignore hole trigger: pending_task={self.pending_task}")
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_task = "hole_wait_insert"

        self.publish_detect_mode("insert")

    def collect_peg_object_frame(self):
        object_idx, base_T_ee = self.parse_peg_trigger_data(self.pending_trigger_msg)
        if self.pending_object_idx is not None:
            object_idx = int(self.pending_object_idx)

        targets = self.make_matrix_targets_from_objects(
            self.latest_objects,
            base_T_ee,
            requested_object_idx=object_idx,
        )

        self.get_logger().info(
            f"[PEG-OBJECT] requested_id={object_idx}, "
            f"targets={len(targets)}, "
            f"visible_ids={self.latest_object_available_ids}, "
            f"perception_status={self.latest_object_status}"
        )

        if not targets:
            self.publish_object_visible_ids_response(
                publisher=self.peg_pub,
                topic_name=self.peg_output_topic,
                visible_ids=self.latest_object_available_ids,
                requested_id=object_idx,
                label="PEG-OBJECT",
            )
            return

        # requested_object_idx already filters the requested class/id.
        # If multiple detections remain, use the highest-confidence target.
        target = max(targets, key=lambda t: float(t["confidence"]))

        self.publish_object_target_once(
            publisher=self.peg_pub,
            topic_name=self.peg_output_topic,
            target=target,
            label="PEG-OBJECT",
        )

    def collect_hole_insert_frame(self):
        base_T_ee = pose6_mm_deg_to_T_mm(self.pending_trigger_msg.data)
        validate_T(base_T_ee, name="base_T_ee")

        # Legacy insert/hole behavior without multi-frame collection
        # and without distance-based duplicate suppression.
        # Publish all valid targets from the current /insert_poses message
        # in [x_mm, y_mm, yaw_deg, id] chunks.
        targets = self.make_legacy_targets_from_inserts(self.latest_inserts, base_T_ee)

        self.get_logger().info(
            f"[HOLE-INSERT] current_frame_targets={len(targets)}"
        )

        if not targets:
            self.get_logger().warn("[HOLE-INSERT] no valid target")
            self.reset_pending()
            return

        self.publish_insert_targets_once(
            publisher=self.hole_pub,
            topic_name=self.hole_output_topic,
            targets=targets,
            label="HOLE-INSERT",
        )

    def publish_object_visible_ids_response(
        self,
        publisher,
        topic_name: str,
        visible_ids: List[int],
        requested_id: Optional[int],
        label: str,
    ):
        msg = Float64MultiArray()
        msg.data = self.object_visible_ids_to_msg_data(visible_ids)

        self.get_logger().warn(
            f"[PUBLISH] topic={topic_name} type={label} format=visible_ids_fallback "
            f"requested_id={requested_id} visible_ids={msg.data} data_len={len(msg.data)} "
            f"rule='len(data)!=17 means requested pose unavailable'"
        )

        self.publish_repeated(publisher, msg, count=10)
        self.reset_pending()

    def publish_object_target_once(self, publisher, topic_name: str, target: Dict[str, Any], label: str):
        msg = Float64MultiArray()
        msg.data = self.object_targets_to_msg_data([target])

        debug_target = {
            "class": target["class"],
            "id": target["id"],
            "confidence": round(float(target["confidence"]), 3),
            "target_frame": target["target_frame"],
            "target_T": np.asarray(target["target_T"]).round(3).tolist(),
        }

        self.get_logger().info(
            f"[PUBLISH] topic={topic_name} type={label} format=matrix16_id_17_success "
            f"target={debug_target} data_len={len(msg.data)}"
        )

        self.publish_repeated(publisher, msg, count=10)
        self.reset_pending()

    def publish_insert_targets_once(self, publisher, topic_name: str, targets: List[Dict[str, Any]], label: str):
        msg = Float64MultiArray()
        msg.data = self.insert_targets_to_msg_data(targets)

        debug_targets = []
        for target in targets:
            debug_targets.append({
                "class": target["class"],
                "id": target["id"],
                "confidence": round(float(target["confidence"]), 3),
                "x": round(float(target["x"]), 3),
                "y": round(float(target["y"]), 3),
                "yaw": round(float(target["yaw"]), 3),
                "yaw_source": target.get("yaw_source", None),
                "source_yaw_deg": target.get("source_yaw_deg", None),
                "source_yaw_score": target.get("source_yaw_score", None),
            })

        self.get_logger().info(
            f"[PUBLISH] topic={topic_name} type={label} format=legacy_xyyaw_4xN "
            f"num_targets={len(targets)} targets={debug_targets} data_len={len(msg.data)}"
        )

        self.publish_repeated(publisher, msg, count=10)
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