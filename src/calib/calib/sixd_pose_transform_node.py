"""
1. 제어부 트리거 (촬영 위치 TCP 좌표)
peg trigger input:
[
  use_cylinder, use_hole, use_cross,
  tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg
]

Object id mapping:
  0: cylinder
  1: hole
  2: cross

The first three values are 0/1 flags.
Objects with flag 0 are excluded.
Among objects with flag 1, the vision node runs 6D pose only for the nearest detected object.

hole trigger input remains unchanged:
[
  tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg
]

2. vision 부에서 정보 구독

peg output:
peg output:
Success, len(data) == 7:
[
  object_id,
  tcp_x_mm,
  tcp_y_mm,
  tcp_z_mm,
  tcp_rx_deg,
  tcp_ry_deg,
  tcp_rz_deg
]

Failure / requested object pose not available, len(data) != 7:
[
  currently_visible_object_id_0,
  currently_visible_object_id_1,
  ...
]

insert / hole output:
[
  x_mm,
  y_mm,
  yaw_deg,
  id
]

좌표 변환 관계 
vision object -> centerd object -> safe centored object (+z축 바닥 방지 , x,y축 변환) -> tcp goal 파지자세 변환

- Final object pose:
      base_T_centered_object =
          base_T_raw_object
          @ raw_object_T_centered_object
      base_T_centered_object_safe =
          defend_centered_object_z_up(base_T_centered_object)
      base_T_tcp_goal =
          base_T_centered_object_safe
          @ centered_object_T_tcp_goal
    output = [object_id, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]

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


# object z축 바닥 방지 함수
def defend_centered_object_z_up(
    T: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
    z_flip_margin: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    object +Z가 world +Z에 더 가깝도록 보정한다.
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    validate_T(T, name="z_defense_in")

    up = normalize_vec(up, name="z_defense_up")
    R = orthonormalize_R(T[:3, :3])

    def _make_right_handed_from_xz(x_raw: np.ndarray, z_raw: np.ndarray, tag: str) -> np.ndarray:
        z = normalize_vec(z_raw, name=f"{tag}_z")

        # x를 z에 수직인 평면으로 투영
        x = np.asarray(x_raw, dtype=np.float64).reshape(3)
        x = x - z * float(np.dot(x, z))
        x = normalize_vec(x, name=f"{tag}_x_proj")

        # 오른손 좌표계 유지: x × y = z 가 되려면 y = z × x
        y = normalize_vec(np.cross(z, x), name=f"{tag}_y")

        return orthonormalize_R(np.column_stack([x, y, z]))

    R_keep = _make_right_handed_from_xz(
        R[:, 0],
        R[:, 2],
        tag="keep",
    )
    keep_dot = float(np.dot(R_keep[:, 2], up))

    R_flip = _make_right_handed_from_xz(
        -R[:, 0],
        -R[:, 2],
        tag="flip",
    )
    flip_dot = float(np.dot(R_flip[:, 2], up))

    z_flipped = flip_dot > keep_dot + float(z_flip_margin)

    if z_flipped:
        T[:3, :3] = R_flip
    else:
        T[:3, :3] = R_keep

    validate_T(T, name="z_defense_out")

    info = {
        "z_flipped": bool(z_flipped),

        "before_dot_up": {
            "x": float(np.dot(R_keep[:, 0], up)),
            "y": float(np.dot(R_keep[:, 1], up)),
            "z": float(keep_dot),
        },

        "keep_dot_up": {
            "x": float(np.dot(R_keep[:, 0], up)),
            "y": float(np.dot(R_keep[:, 1], up)),
            "z": float(keep_dot),
        },
        "flip_dot_up": {
            "x": float(np.dot(R_flip[:, 0], up)),
            "y": float(np.dot(R_flip[:, 1], up)),
            "z": float(flip_dot),
        },

        "after_dot_up": {
            "x": float(np.dot(T[:3, 0], up)),
            "y": float(np.dot(T[:3, 1], up)),
            "z": float(np.dot(T[:3, 2], up)),
        },
    }

    return T, info

# center/object frame의 X/Y 축 중 월드 XY 평면에 더 평행한 축을 새 object +X로 선택

def canonicalize_xy_flatter_as_x(
    T: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
    swap_margin: float = 0.03,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    validate_T(T, name="xy_flatten_in")

    up = normalize_vec(up, name="xy_flatten_up")
    R = orthonormalize_R(T[:3, :3])

    x_old = normalize_vec(R[:, 0], "xy_x_old")
    y_old = normalize_vec(R[:, 1], "xy_y_old")
    z = normalize_vec(R[:, 2], "xy_z")

    x_flatness = abs(float(np.dot(x_old, up)))
    y_flatness = abs(float(np.dot(y_old, up)))

    # y축이 x축보다 충분히 더 수평이면 y를 새 x로 채택
    swapped_xy = y_flatness + float(swap_margin) < x_flatness

    if swapped_xy:
        x_new = y_old.copy()
        # 오른손 좌표계 유지: y = z × x
        y_new = normalize_vec(np.cross(z, x_new), "xy_y_new")
    else:
        x_new = x_old.copy()
        y_new = normalize_vec(np.cross(z, x_new), "xy_y_keep")

    # 수치오차 방지: x를 z에 수직으로 재투영 후 y 재계산
    x_new = normalize_vec(x_new - z * float(np.dot(x_new, z)), "xy_x_proj")
    y_new = normalize_vec(np.cross(z, x_new), "xy_y_final")

    T[:3, :3] = orthonormalize_R(np.column_stack([x_new, y_new, z]))
    validate_T(T, name="xy_flatten_out")

    info = {
        "xy_swapped": bool(swapped_xy),
        "x_flatness_before": float(x_flatness),
        "y_flatness_before": float(y_flatness),
        "selected_x_from": "old_y" if swapped_xy else "old_x",
        "x_flatness_after": abs(float(np.dot(T[:3, 0], up))),
        "y_flatness_after": abs(float(np.dot(T[:3, 1], up))),
        "z_dot_up_after": float(np.dot(T[:3, 2], up)),
    }
    return T, info

# json을 4x4 mat로 변환하는 util
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


def wrap_deg(a: float) -> float:
    """각도를 [-180, 180) 범위로 정규화."""
    return float((float(a) + 180.0) % 360.0 - 180.0)

def T_rot_z_deg(deg: float) -> np.ndarray:
    """object/center local +Z 기준 yaw 회전 4x4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(0.0, 0.0, float(deg))
    return T

def signed_angle_about_axis_deg(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """axis 기준 v_from -> v_to signed angle [deg]."""
    axis = normalize_vec(axis, name="signed_angle_axis")
    a = np.asarray(v_from, dtype=np.float64).reshape(3)
    b = np.asarray(v_to, dtype=np.float64).reshape(3)

    # axis에 수직인 평면으로 projection
    a = a - axis * float(np.dot(a, axis))
    b = b - axis * float(np.dot(b, axis))
    a = normalize_vec(a, name="signed_angle_from_proj")
    b = normalize_vec(b, name="signed_angle_to_proj")

    s = float(np.dot(axis, np.cross(a, b)))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arctan2(s, c)))


# ============================================================
# rbpodo TCP 쿼리 헬퍼 (handeye_sampler 코드 참고)
# ============================================================


class ObjectPoseTransformNode(Node):
    """
    Mixed output transform node.

    Input:
        /manipulation/trigger_peg:
            std_msgs/Float64MultiArray
            data = [use_cylinder, use_hole, use_cross, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            The first three values are 0/1 flags for object ids:
                0: cylinder, 1: hole, 2: cross.
            Objects with flag 0 are excluded.
            The allowed ids are relayed to the 6D pose trigger topic.

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
            Success: 7 floats:
                [selected_object_id, tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg]

            Failure / requested/allowed object pose not available: variable-length id list:
                [currently_visible_object_id_0, currently_visible_object_id_1, ...]

            Therefore the controller can branch by len(data):
                len(data) == 7 -> success [object_id, moveL pose6] response
                len(data) != 7 -> failure/fallback visible-id response

            The published pose6 part, data[1:7], is intended to be sent directly to robot moveL.

        /vision/hole_targets:
            insert target, legacy format. 4 floats per target:
            [x_mm, y_mm, yaw_deg, id]
    """

    def __init__(self):
        super().__init__("sixd_pose_transform_node")

        # ----------------------------
        # Parameters
        self.declare_parameter("handeye_result_path", "")
        self.declare_parameter("object_grasp_yaml_path", "")

        # Final spec:
        #   raw object -> YAML centered object -> Z-up defense -> TCP grasp -> pose6.
        # Z defense only flips +Z/+X when centered object +Z points toward world -Z.
        # It does NOT swap X/Y because object X is used as the gripper alignment axis.
        self.declare_parameter("canonicalize_object_axes", True)
        self.declare_parameter("canonicalize_z_flip_margin", 0.05)

        # X/Y 중 더 수평인 축을 object +X로 canonicalize.
        self.declare_parameter("canonicalize_xy_flatter_as_x", True)
        self.declare_parameter("canonicalize_xy_swap_margin", 0.03)
        self.declare_parameter("canonicalize_xy_max_flatness", 0.85)

        # Symmetry yaw candidates 중 RB5 마지막 joint 제한 고려 기준으로 안전한 grasp 후보 선택
        self.declare_parameter("reference_last_joint_deg", 34.16)
        self.declare_parameter("last_joint_limit_delta_deg", 95.0)

        self.declare_parameter("min_confidence", 0.3)

        # topic names
        self.declare_parameter("object_topic", "/object_poses")
        self.declare_parameter("insert_topic", "/insert_poses")
        self.declare_parameter("detect_mode_topic", "/detect_mode")
        self.declare_parameter("peg_trigger_topic", "/manipulation/trigger_peg")
        self.declare_parameter("hole_trigger_topic", "/manipulation/trigger_hole")
        self.declare_parameter("object_6d_trigger_topic", "/manipulation/object_6d_trigger")
        self.declare_parameter("peg_output_topic", "/vision/peg_targets")
        self.declare_parameter("hole_output_topic", "/vision/hole_targets")


        self.declare_parameter("detect_mode_settle_sec", 0.5)

        # matplt debug
        self.declare_parameter("visualize_pose6_target", True)
        self.declare_parameter("visualize_axes_length_mm", 50.0)
        self.declare_parameter("visualize_approach_length_mm", 80.0)
        self.declare_parameter("visualize_blocking", False)
        self.declare_parameter("visualize_save_dir", "")

        self._viz_fig = None
        self._viz_ax = None

        self.class_to_id = {
            "cylinder": 0,
            "cylinder_insert": 0,
            "hole": 1,
            "hole_insert": 1,
            "cross": 2,
            "cross_insert": 2,
        }
        self.id_to_class = {
            -1: "nearest",
            0: "cylinder",
            1: "hole",
            2: "cross",
        }

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.reference_last_joint_deg = float(self.get_parameter("reference_last_joint_deg").value)
        self.last_joint_limit_delta_deg = float(self.get_parameter("last_joint_limit_delta_deg").value)

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

        self.object_trigger_delay_timer = None
        self.insert_settle_timer = None

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
            "peg trigger=/manipulation/trigger_peg: [use_cylinder, use_hole, use_cross, tcp_pose6]; "
            "object output=/vision/peg_targets: success len=7 [selected_object_id, tcp_moveL_pose6], failure len!=7 [visible_ids]; "
            "insert output=/vision/hole_targets: [x_mm, y_mm, yaw_deg, id]."
        )

    # ============================================================
    # State helpers
    # ============================================================

    def cancel_timer_if_alive(self, timer_attr: str):
        timer = getattr(self, timer_attr, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
            setattr(self, timer_attr, None)

    def reset_pending(self):
        self.cancel_timer_if_alive("object_trigger_delay_timer")
        self.cancel_timer_if_alive("insert_settle_timer")
        self.pending_task = None
        self.pending_trigger_msg = None
        self.pending_object_idx = None

    def publish_detect_mode(self, mode):
        msg = String()
        msg.data = mode
        self.detect_mode_pub.publish(msg)
        self.get_logger().info(f"request detect_mode: {mode}")

    def publish_object_6d_trigger(self, allowed_object_ids: List[int]):
        allowed_object_ids = [int(v) for v in allowed_object_ids]
        allowed_classes = [self.class_name_from_id(v) for v in allowed_object_ids]

        msg = String()
        msg.data = json.dumps(
            {
                "allowed_ids": allowed_object_ids,
                "allowed_classes": allowed_classes,
            },
            ensure_ascii=False,
        )
        self.object_6d_trigger_pub.publish(msg)

        self.get_logger().info(
            f"request object 6D pose: topic={self.object_6d_trigger_topic}, "
            f"allowed_ids={allowed_object_ids}, allowed_classes={allowed_classes}"
        )
        return allowed_classes

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
        Load object-frame YAML.

        Preferred YAML format after final spec change:

        unit: mm
        objects:
          cylinder:
            # RAW/CAD object frame -> centered/canonical object frame.
            # Legacy key object_to_grasp is accepted as the same meaning.
            object_to_center:
              unit: mm
              matrix:
                - [1, 0, 0, 0]
                - [0, 1, 0, 0]
                - [0, 0, 1, 0]
                - [0, 0, 0, 1]

            # Centered object frame -> final robot TCP frame for moveL.
            # If omitted, identity is used, meaning object_to_center already
            # defines the final TCP frame.
            centered_object_to_tcp:
              unit: mm
              matrix:
                - [1, 0, 0, 0]
                - [0, 1, 0, 0]
                - [0, 0, 1, 0]
                - [0, 0, 0, 1]
        """
        yaml_path_str = str(self.get_parameter("object_grasp_yaml_path").value)
        cfg: Dict[str, Dict[str, Any]] = {}

        if not yaml_path_str:
            self.get_logger().warn(
                "object_grasp_yaml_path is empty. "
                "Using identity raw_object_T_centered_object and centered_object_T_tcp_goal."
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

        def _load_T_from_keys(obj_data: Dict[str, Any], keys: List[str], default_T: np.ndarray) -> np.ndarray:
            for key in keys:
                block = obj_data.get(key, None)
                if isinstance(block, dict):
                    unit = block.get("unit", obj_data.get("unit", global_unit))
                    if "matrix" in block:
                        return matrix_from_data(block["matrix"], unit=unit)
                elif block is not None:
                    # Allow direct 4x4 matrix under the key.
                    unit = obj_data.get("unit", global_unit)
                    return matrix_from_data(block, unit=unit)

            return default_T.copy()

        for name, obj_data in objects.items():
            if obj_data is None:
                obj_data = {}
            if not isinstance(obj_data, dict):
                raise ValueError(f"object_grasp_yaml: objects.{name} must be a dict.")

            # Backward compatibility:
            #   object_to_grasp no longer means direct grasp pose.
            #   It is treated as RAW/CAD object -> centered/canonical object frame.
            raw_object_T_centered_object = _load_T_from_keys(
                obj_data,
                keys=["object_to_center", "object_to_canonical", "object_to_grasp"],
                default_T=np.eye(4, dtype=np.float64),
            )

            # Final optional TCP offset from the centered object frame.
            centered_object_T_tcp_goal = _load_T_from_keys(
                obj_data,
                keys=["centered_object_to_tcp", "center_to_tcp", "canonical_to_tcp", "object_center_to_tcp"],
                default_T=np.eye(4, dtype=np.float64),
            )

            cfg[str(name)] = {
                "raw_object_T_centered_object": raw_object_T_centered_object,
                "centered_object_T_tcp_goal": centered_object_T_tcp_goal,
                "symmetry": obj_data.get("symmetry", {}) or {},
            }

        self.get_logger().info(f"loaded object target YAML: {path}")
        return cfg

    def get_object_target_transforms_for_class(self, cls: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        base_name = canonical_object_name(cls)

        if cls in self.grasp_cfg:
            item = self.grasp_cfg[cls]
        elif base_name in self.grasp_cfg:
            item = self.grasp_cfg[base_name]
        else:
            item = {
                "raw_object_T_centered_object": np.eye(4, dtype=np.float64),
                "centered_object_T_tcp_goal": np.eye(4, dtype=np.float64),
                "symmetry": {},
            }

        raw_object_T_centered_object = np.asarray(
            item["raw_object_T_centered_object"], dtype=np.float64
        ).reshape(4, 4)
        centered_object_T_tcp_goal = np.asarray(
            item["centered_object_T_tcp_goal"], dtype=np.float64
        ).reshape(4, 4)
        symmetry = item.get("symmetry", {}) or {}

        validate_T(raw_object_T_centered_object, name=f"raw_object_T_centered_object[{cls}]")
        validate_T(centered_object_T_tcp_goal, name=f"centered_object_T_tcp_goal[{cls}]")

        return raw_object_T_centered_object.copy(), centered_object_T_tcp_goal.copy(), dict(symmetry)

    def make_symmetric_grasp_candidates(
        self,
        centered_object_T_tcp_nominal: np.ndarray,
        symmetry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        centered_object_to_tcp 기본 grasp를 centered object local +Z 기준 yaw 후보로 회전시킨다.

        YAML 예:
            symmetry:
              axis: "+z"
              yaw_candidates_deg: [0.0, 180.0]
        """
        yaw_candidates = symmetry.get("yaw_candidates_deg", [0.0])
        candidates: List[Dict[str, Any]] = []

        for yaw in yaw_candidates:
            yaw = float(yaw)
            centered_object_T_tcp = T_rot_z_deg(yaw) @ centered_object_T_tcp_nominal
            validate_T(centered_object_T_tcp, name=f"centered_object_T_tcp_candidate[yaw={yaw}]")
            candidates.append({
                "name": f"yaw_{yaw:.0f}",
                "yaw_deg": yaw,
                "centered_object_T_tcp": centered_object_T_tcp,
            })

        if not candidates:
            candidates.append({
                "name": "yaw_0",
                "yaw_deg": 0.0,
                "centered_object_T_tcp": centered_object_T_tcp_nominal.copy(),
            })

        return candidates

    def estimate_last_joint_for_goal(
        self,
        base_T_goal: np.ndarray,
        reference_T_tcp: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        목표 TCP 자세가 기준 TCP 자세에서 마지막 joint를 얼마나 더 돌려야 하는지 근사 추정한다.

        trigger로 받은 현재 TCP pose6를 reference_T_tcp로 사용한다.
        전체 IK를 푸는 것이 아니라, TCP +X 방향 변화를 TCP +Y축 기준 signed angle로 보고
        reference_last_joint_deg에 더한 값을 estimated J5로 사용한다.
        """
        R_ref = orthonormalize_R(reference_T_tcp[:3, :3])
        R_goal = orthonormalize_R(base_T_goal[:3, :3])

        ref_x = R_ref[:, 0]
        goal_x = R_goal[:, 0]
        goal_y = R_goal[:, 1]

        signed_delta = signed_angle_about_axis_deg(ref_x, goal_x, goal_y)
        estimated_j5 = wrap_deg(self.reference_last_joint_deg + signed_delta)
        delta_abs = abs(wrap_deg(estimated_j5 - self.reference_last_joint_deg))
        return estimated_j5, delta_abs, signed_delta

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
    # Transform core - object output: final moveL pose6
    # ============================================================

    def object_to_pose6_target_dict(
        self,
        obj: Dict[str, Any],
        base_T_ee: np.ndarray,
        allowed_object_ids: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        cls = str(obj.get("class", ""))
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            return None

        if cls not in self.class_to_id:
            return None

        obj_id = int(self.class_to_id[cls])
        if allowed_object_ids is not None:
            allowed_object_ids = [int(v) for v in allowed_object_ids]
            if obj_id not in allowed_object_ids:
                return None

        try:
            # FoundationPose/CAD raw object pose in camera frame.
            cam_T_obj = object_json_to_cam_T_obj_mm(obj)

            # Trigger에서 받은 현재 TCP/EE pose + hand-eye + camera object pose.
            # 1번 코드처럼 rbpodo로 직접 TCP를 다시 쿼리하지 않고, 2번의 기존 trigger TCP를 유지한다.
            base_T_obj_raw = base_T_ee @ self.ee_T_cam @ cam_T_obj
            validate_T(base_T_obj_raw, name=f"base_T_obj_raw[{cls}]")

            # YAML:
            #   raw object/CAD frame -> centered/canonical object frame
            #   centered object frame -> nominal TCP goal frame
            raw_object_T_centered_object, centered_object_T_tcp_nominal, symmetry = (
                self.get_object_target_transforms_for_class(cls)
            )

            base_T_centered_object = base_T_obj_raw @ raw_object_T_centered_object
            validate_T(base_T_centered_object, name=f"base_T_centered_object[{cls}]")

            # 1번 최신 Z-up defense:
            # keep 후보와 X/Z 동시 flip 후보 중 object +Z가 world +Z에 더 가까운 쪽 선택.
            axis_info = None
            if bool(self.get_parameter("canonicalize_object_axes").value):
                base_T_centered_object, axis_info = defend_centered_object_z_up(
                    base_T_centered_object,
                    z_flip_margin=float(self.get_parameter("canonicalize_z_flip_margin").value),
                )
                if axis_info["z_flipped"]:
                    self.get_logger().info(
                        f"z-defense centered object axes class={cls} "
                        f"z_flipped={axis_info['z_flipped']} "
                        f"keep_dot_up={axis_info.get('keep_dot_up')} "
                        f"flip_dot_up={axis_info.get('flip_dot_up')} "
                        f"after_dot_up={axis_info['after_dot_up']}"
                    )

            # 1번 전체 로직: X/Y 중 월드 XY 평면에 더 평행한 축을 object +X로 선택.
            # 사용자가 A안을 선택했으므로 기본값 True.
            xy_info = None
            if bool(self.get_parameter("canonicalize_xy_flatter_as_x").value):
                base_T_centered_object, xy_info = canonicalize_xy_flatter_as_x(
                    base_T_centered_object,
                    swap_margin=float(self.get_parameter("canonicalize_xy_swap_margin").value),
                )
                self.get_logger().info(
                    f"[XY_FLAT] cls={cls} "
                    f"swapped={xy_info['xy_swapped']} "
                    f"selected_x={xy_info['selected_x_from']} "
                    f"x_flat_before={xy_info['x_flatness_before']:.3f} "
                    f"y_flat_before={xy_info['y_flatness_before']:.3f} "
                    f"x_flat_after={xy_info['x_flatness_after']:.3f}"
                )

                max_flat = float(self.get_parameter("canonicalize_xy_max_flatness").value)
                if xy_info["x_flatness_after"] > max_flat:
                    self.get_logger().warn(
                        f"[SKIP] cls={cls}: selected object +X is still too steep. "
                        f"x_flatness_after={xy_info['x_flatness_after']:.3f} > {max_flat:.3f}"
                    )
                    return None

            # symmetry.yaw_candidates_deg 후보를 만들고, RB5 마지막 joint 기준으로 필터링.
            grasp_candidates = self.make_symmetric_grasp_candidates(
                centered_object_T_tcp_nominal,
                symmetry,
            )

            best = None
            for cand in grasp_candidates:
                base_T_tcp_goal_cand = base_T_centered_object @ cand["centered_object_T_tcp"]
                validate_T(base_T_tcp_goal_cand, name=f"base_T_tcp_goal[{cls}:{cand['name']}]")

                estimated_j5, delta_j5, signed_delta_j5 = self.estimate_last_joint_for_goal(
                    base_T_tcp_goal_cand,
                    base_T_ee,
                )

                if delta_j5 > self.last_joint_limit_delta_deg + 1e-9:
                    self.get_logger().info(
                        f"[J5_SKIP] cls={cls} cand={cand['name']} "
                        f"yaw={cand['yaw_deg']:.1f} "
                        f"est_J5={estimated_j5:.2f} "
                        f"signed_delta={signed_delta_j5:.2f} "
                        f"delta={delta_j5:.2f} "
                        f"limit=±{self.last_joint_limit_delta_deg:.1f}"
                    )
                    continue

                score = delta_j5
                if best is None or score < best["score"]:
                    best = {
                        "candidate": cand,
                        "base_T_tcp_goal": base_T_tcp_goal_cand,
                        "estimated_j5": estimated_j5,
                        "delta_j5": delta_j5,
                        "signed_delta_j5": signed_delta_j5,
                        "score": score,
                    }

                self.get_logger().info(
                    f"[J5_CAND] cls={cls} cand={cand['name']} "
                    f"yaw={cand['yaw_deg']:.1f} "
                    f"est_J5={estimated_j5:.2f} "
                    f"signed_delta={signed_delta_j5:.2f} "
                    f"delta={delta_j5:.2f} "
                    f"score={score:.2f}"
                )

            if best is None:
                self.get_logger().warn(
                    f"[NO_VALID_GRASP] cls={cls}: all yaw candidates exceed "
                    f"J5 reference {self.reference_last_joint_deg:.2f} "
                    f"±{self.last_joint_limit_delta_deg:.1f} deg"
                )
                return None

            selected = best["candidate"]
            base_T_tcp_goal = best["base_T_tcp_goal"]
            target_pose6 = T_mm_to_pose6_mm_deg(base_T_tcp_goal)
            p = base_T_tcp_goal[:3, 3]

            return {
                "class": cls,
                "id": obj_id,
                "confidence": conf,
                "source_depth_m": self.object_source_depth_m(obj),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "target_T": base_T_tcp_goal,
                "target_pose6": target_pose6,
                "base_T_obj_raw": base_T_obj_raw,
                "base_T_centered_object_safe": base_T_centered_object,
                "target_frame": "tcp_goal",
                "axis_info": axis_info,
                "xy_info": xy_info,
                "selected_grasp_name": selected["name"],
                "selected_grasp_yaw_deg": float(selected["yaw_deg"]),
                "estimated_last_joint_deg": float(best["estimated_j5"]),
                "last_joint_delta_deg": float(best["delta_j5"]),
                "last_joint_signed_delta_deg": float(best["signed_delta_j5"]),
                "output_format": "id_plus_moveL_pose6_7",
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

    @staticmethod
    def object_source_depth_m(obj: Dict[str, Any]) -> float:
        """Depth median from vision JSON, used only for robust nearest fallback."""
        try:
            priority = obj.get("priority", {})
            if isinstance(priority, dict) and "depth_median_m" in priority:
                return float(priority["depth_median_m"])
            if "depth_median_m" in obj:
                return float(obj["depth_median_m"])
        except Exception:
            pass
        return float("inf")

    def make_pose6_targets_from_objects(self, objects, base_T_ee, allowed_object_ids=None):
        targets = []
        for obj in objects:
            target = self.object_to_pose6_target_dict(
                obj,
                base_T_ee,
                allowed_object_ids=allowed_object_ids,
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
        Object target success format:
            [object_id, tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg]

        object_id is the actual recognized object id from vision result class.
        Total 7 floats per target.
        """
        data = []

        for t in targets:
            obj_id = float(t["id"])
            pose6 = np.asarray(t["target_pose6"], dtype=np.float64).reshape(6)
            data.extend([obj_id] + pose6.tolist())

        return data

    @staticmethod
    def object_visible_ids_to_msg_data(visible_ids):
        """
        Peg fallback/failure format:
            [currently_visible_object_id_0, currently_visible_object_id_1, ...]

        The controller distinguishes this from success by length:
            len(data) == 7 -> success [object_id, moveL pose6]
            len(data) != 7 -> fallback visible-id response
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
    def parse_peg_trigger_data(trigger_msg) -> Tuple[List[int], np.ndarray]:
        data = np.asarray(trigger_msg.data, dtype=np.float64).reshape(-1)
        if data.size < 9:
            raise ValueError(
                "peg trigger data must be "
                "[use_cylinder, use_hole, use_cross, "
                "x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]. "
                f"got {data.size} values"
            )

        use_flags = [int(round(float(v))) for v in data[:3]]
        for flag in use_flags:
            if flag not in (0, 1):
                raise ValueError(
                    "peg trigger use flags must be 0 or 1: "
                    f"[use_cylinder, use_hole, use_cross]={use_flags}"
                )

        allowed_object_ids = [obj_id for obj_id, flag in enumerate(use_flags) if flag == 1]
        if not allowed_object_ids:
            raise ValueError(
                "peg trigger must enable at least one object. "
                "example: [1, 0, 1, tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz]"
            )

        base_T_ee = pose6_mm_deg_to_T_mm(data[3:9])
        validate_T(base_T_ee, name="base_T_ee")
        return allowed_object_ids, base_T_ee

    def schedule_once(self, delay_sec: float, timer_attr: str, callback):
        delay_sec = max(float(delay_sec), 0.0)

        self.cancel_timer_if_alive(timer_attr)

        if delay_sec <= 1e-6:
            callback()
            return

        def _wrapped_callback():
            self.cancel_timer_if_alive(timer_attr)
            callback()

        setattr(self, timer_attr, self.create_timer(delay_sec, _wrapped_callback))

    def peg_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(f"ignore peg trigger: pending_task={self.pending_task}")
            return

        try:
            allowed_object_ids, _ = self.parse_peg_trigger_data(trigger_msg)
            for object_idx in allowed_object_ids:
                self.class_name_from_id(object_idx)
        except Exception as e:
            self.get_logger().warn(f"invalid peg trigger: {e}")
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_object_idx = allowed_object_ids
        self.pending_task = "peg_settling"

        self.publish_detect_mode("object")

        delay_sec = float(self.get_parameter("detect_mode_settle_sec").value)
        self.get_logger().info(
            f"peg trigger accepted: allowed_ids={allowed_object_ids}, "
            f"switch detect_mode=object, wait {delay_sec:.3f}s, "
            f"then send object 6D trigger"
        )
        self.schedule_once(
            delay_sec,
            "object_trigger_delay_timer",
            lambda: self.finish_peg_settle_and_trigger_object(allowed_object_ids),
        )

    def finish_peg_settle_and_trigger_object(self, allowed_object_ids: List[int]):
        if self.pending_task != "peg_settling":
            self.get_logger().warn(
                f"skip delayed object 6D trigger: pending_task={self.pending_task}"
            )
            return

        self.pending_task = "peg_wait_object"
        self.publish_object_6d_trigger(allowed_object_ids)

    def hole_trigger_callback(self, trigger_msg):
        if self.pending_task is not None:
            self.get_logger().warn(f"ignore hole trigger: pending_task={self.pending_task}")
            return

        self.pending_trigger_msg = trigger_msg
        self.pending_task = "hole_settling"

        self.publish_detect_mode("insert")

        delay_sec = float(self.get_parameter("detect_mode_settle_sec").value)
        self.get_logger().info(
            f"hole trigger accepted: switch detect_mode=insert, "
            f"wait {delay_sec:.3f}s, then accept next insert result"
        )
        self.schedule_once(
            delay_sec,
            "insert_settle_timer",
            self.finish_hole_settle_and_wait_insert,
        )

    def finish_hole_settle_and_wait_insert(self):
        if self.pending_task != "hole_settling":
            self.get_logger().warn(
                f"skip insert wait after settling: pending_task={self.pending_task}"
            )
            return

        self.pending_task = "hole_wait_insert"
        self.get_logger().info("hole trigger: now accepting next /insert_poses message")

    def collect_peg_object_frame(self):
        allowed_object_ids, base_T_ee = self.parse_peg_trigger_data(self.pending_trigger_msg)
        if self.pending_object_idx is not None:
            allowed_object_ids = [int(v) for v in self.pending_object_idx]

        targets = self.make_pose6_targets_from_objects(
            self.latest_objects,
            base_T_ee,
            allowed_object_ids=allowed_object_ids,
        )

        self.get_logger().info(
            f"[PEG-OBJECT] allowed_ids={allowed_object_ids}, "
            f"targets={len(targets)}, "
            f"visible_ids={self.latest_object_available_ids}, "
            f"perception_status={self.latest_object_status}"
        )

        if not targets:
            self.publish_object_visible_ids_response(
                publisher=self.peg_pub,
                topic_name=self.peg_output_topic,
                visible_ids=self.latest_object_available_ids,
                requested_id=None,
                label="PEG-OBJECT",
            )
            return

        # Vision normally returns one nearest allowed object.
        # If multiple objects are received for backward compatibility, choose nearest depth first,
        # then higher confidence as a tie-breaker.
        target = min(
            targets,
            key=lambda t: (
                float(t.get("source_depth_m", float("inf"))),
                -float(t.get("confidence", 0.0)),
            ),
        )

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

    # ============================================================
    # Visualization - final moveL pose6 target
    # ============================================================

    @staticmethod
    def set_axes_equal_3d(ax):
        """
        Make 3D axis scales equal so pose directions are not visually distorted.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range, 1.0])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def draw_frame_3d(ax, T: np.ndarray, name: str, axis_len: float, alpha: float = 1.0):
        """
        Draw a coordinate frame in base/world coordinates.
        Columns of R are the frame X/Y/Z axes expressed in base/world.
        """
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        p = T[:3, 3]
        R = orthonormalize_R(T[:3, :3])

        # X/Y/Z axis colors are conventional for debug visualization.
        colors = ["r", "g", "b"]
        labels = [f"{name} +X", f"{name} +Y", f"{name} +Z"]

        for i in range(3):
            v = R[:, i] * float(axis_len)
            ax.quiver(
                p[0], p[1], p[2],
                v[0], v[1], v[2],
                color=colors[i], alpha=alpha,
                arrow_length_ratio=0.18,
                linewidth=1.6,
            )

        ax.text(p[0], p[1], p[2], f" {name}")

    def visualize_pose6_target_once(self, target: Dict[str, Any]):
        """
        Show one 3D preview for the final moveL pose6 target.

        Important convention from the old hand-eye sampler:
            TCP/local -Y is the look/approach direction.

        Therefore this preview draws:
            - TCP frame axes
            - a thick magenta arrow along TCP -Y: where the TCP target is looking
            - a dashed pregrasp segment from +TCP_Y toward the target
        """
        if not bool(self.get_parameter("visualize_pose6_target").value):
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            self.get_logger().warn(f"visualize_pose6_target disabled: matplotlib import failed: {e}")
            return

        try:
            axis_len = float(self.get_parameter("visualize_axes_length_mm").value)
            approach_len = float(self.get_parameter("visualize_approach_length_mm").value)
            blocking = bool(self.get_parameter("visualize_blocking").value)
            save_dir = str(self.get_parameter("visualize_save_dir").value).strip()

            T_tcp = np.asarray(target["target_T"], dtype=np.float64).reshape(4, 4)
            pose6 = np.asarray(target["target_pose6"], dtype=np.float64).reshape(6)
            T_center = np.asarray(target.get("base_T_centered_object_safe", T_tcp), dtype=np.float64).reshape(4, 4)
            T_raw = np.asarray(target.get("base_T_obj_raw", T_center), dtype=np.float64).reshape(4, 4)

            validate_T(T_tcp, name="viz T_tcp")
            validate_T(T_center, name="viz T_center")
            validate_T(T_raw, name="viz T_raw")

            p_tcp = T_tcp[:3, 3]
            p_center = T_center[:3, 3]
            p_raw = T_raw[:3, 3]
            R_tcp = orthonormalize_R(T_tcp[:3, :3])

            # Same convention as old hand-eye code:
            #     camera/tool forward = local -Y
            tcp_look_dir = -R_tcp[:, 1]
            tcp_plus_y = R_tcp[:, 1]
            p_pre = p_tcp + tcp_plus_y * approach_len

            plt.ion()
            if self._viz_fig is None or self._viz_ax is None:
                self._viz_fig = plt.figure("moveL pose6 target preview")
                self._viz_ax = self._viz_fig.add_subplot(111, projection="3d")
            else:
                self._viz_ax.clear()

            ax = self._viz_ax
            ax.set_title(
                f"moveL pose6 target | class={target.get('class')} id={target.get('id')}\n"
                f"TCP -Y is look/approach direction | pose6={np.round(pose6, 2).tolist()}"
            )
            ax.set_xlabel("Base X [mm]")
            ax.set_ylabel("Base Y [mm]")
            ax.set_zlabel("Base Z [mm]")

            # World/base frame near the target area for orientation reference.
            T_base_ref = np.eye(4, dtype=np.float64)
            T_base_ref[:3, 3] = p_center
            self.draw_frame_3d(ax, T_base_ref, "base ref", axis_len * 0.8, alpha=0.35)

            self.draw_frame_3d(ax, T_raw, "raw obj", axis_len * 0.75, alpha=0.35)
            self.draw_frame_3d(ax, T_center, "center obj", axis_len, alpha=0.75)
            self.draw_frame_3d(ax, T_tcp, "TCP goal", axis_len * 1.2, alpha=1.0)

            # Points and relation lines.
            ax.scatter([p_raw[0]], [p_raw[1]], [p_raw[2]], marker="o", s=35, label="raw object origin")
            ax.scatter([p_center[0]], [p_center[1]], [p_center[2]], marker="^", s=55, label="centered object origin")
            ax.scatter([p_tcp[0]], [p_tcp[1]], [p_tcp[2]], marker="*", s=120, label="TCP goal origin")

            ax.plot(
                [p_center[0], p_tcp[0]],
                [p_center[1], p_tcp[1]],
                [p_center[2], p_tcp[2]],
                linestyle="--", linewidth=1.2, label="center -> TCP offset"
            )

            # TCP look/approach direction: local -Y.
            v = tcp_look_dir * approach_len
            ax.quiver(
                p_tcp[0], p_tcp[1], p_tcp[2],
                v[0], v[1], v[2],
                color="m", linewidth=3.0,
                arrow_length_ratio=0.20,
            )
            ax.text(
                p_tcp[0] + v[0], p_tcp[1] + v[1], p_tcp[2] + v[2],
                " TCP look/approach (-Y)",
                color="m",
            )

            # Pregrasp position if controller approaches along -TCP_Y.
            ax.scatter([p_pre[0]], [p_pre[1]], [p_pre[2]], marker="x", s=80, label="pregrasp = TCP +Y")
            ax.plot(
                [p_pre[0], p_tcp[0]],
                [p_pre[1], p_tcp[1]],
                [p_pre[2], p_tcp[2]],
                linestyle=":", linewidth=2.0, label="pregrasp move direction"
            )

            # Make bounds around all relevant points.
            pts = np.vstack([
                p_raw.reshape(1, 3),
                p_center.reshape(1, 3),
                p_tcp.reshape(1, 3),
                p_pre.reshape(1, 3),
                (p_tcp + tcp_look_dir * approach_len).reshape(1, 3),
            ])
            margin = max(axis_len, approach_len) * 1.4
            mins = pts.min(axis=0) - margin
            maxs = pts.max(axis=0) + margin
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            self.set_axes_equal_3d(ax)
            ax.legend(loc="upper left", fontsize=8)

            self._viz_fig.tight_layout()

            if save_dir:
                out_dir = Path(save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"pose6_target_{target.get('class')}_{target.get('id')}.png"
                self._viz_fig.savefig(str(out_path), dpi=150)
                self.get_logger().info(f"saved pose6 target preview: {out_path}")

            plt.show(block=blocking)
            plt.pause(0.001)

        except Exception as e:
            self.get_logger().warn(f"failed to visualize pose6 target: {e}")

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
            f"rule='len(data)!=7 means requested pose unavailable'"
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
            "source_depth_m": target.get("source_depth_m", None),
            "target_frame": target["target_frame"],
            "selected_grasp": target.get("selected_grasp_name", None),
            "selected_yaw_deg": target.get("selected_grasp_yaw_deg", None),
            "estimated_last_joint_deg": target.get("estimated_last_joint_deg", None),
            "last_joint_delta_deg": target.get("last_joint_delta_deg", None),
            "target_pose6": np.asarray(target["target_pose6"]).round(3).tolist(),
            "target_T": np.asarray(target["target_T"]).round(3).tolist(),
        }

        self.get_logger().info(
            f"[PUBLISH] topic={topic_name} type={label} format=id_plus_moveL_pose6_7_success "
            f"target={debug_target} data_len={len(msg.data)}"
        )

        self.visualize_pose6_target_once(target)

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