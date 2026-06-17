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
    output = [object_id, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] or [-object_id, pose6..., empty_x_mm, empty_y_mm]

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


def compute_centered_axis_tilt_info(
    T: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
) -> Dict[str, Any]:
    """
    centered/safe object frame의 각 축이 world XY 평면 또는 world +Z와 이루는 각도를 계산한다.

    x_tilt_ground_deg / y_tilt_ground_deg:
        각 축이 world XY 평면에서 얼마나 들렸는지 [deg].
        0 deg이면 완전히 XY 평면에 평행하고, 90 deg이면 world Z 방향이다.

    z_tilt_up_deg:
        object +Z가 world +Z에서 얼마나 기울었는지 [deg].
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    validate_T(T, name="compute_centered_axis_tilt_info")

    up = normalize_vec(up, name="tilt_info_up")
    R = orthonormalize_R(T[:3, :3])
    p = T[:3, 3]

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    x_dot_up = float(np.dot(x_axis, up))
    y_dot_up = float(np.dot(y_axis, up))
    z_dot_up = float(np.dot(z_axis, up))

    x_tilt_ground_deg = float(
        np.degrees(np.arcsin(np.clip(abs(x_dot_up), 0.0, 1.0)))
    )
    y_tilt_ground_deg = float(
        np.degrees(np.arcsin(np.clip(abs(y_dot_up), 0.0, 1.0)))
    )
    z_tilt_up_deg = float(
        np.degrees(np.arccos(np.clip(z_dot_up, -1.0, 1.0)))
    )

    return {
        "p": p,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "x_dot_up": x_dot_up,
        "y_dot_up": y_dot_up,
        "z_dot_up": z_dot_up,
        "x_tilt_ground_deg": x_tilt_ground_deg,
        "y_tilt_ground_deg": y_tilt_ground_deg,
        "z_tilt_up_deg": z_tilt_up_deg,
    }


def log_centered_axis_tilt(logger, label: str, cls: str, tilt_info: Dict[str, Any]):
    p = tilt_info["p"]
    x_axis = tilt_info["x_axis"]
    y_axis = tilt_info["y_axis"]
    z_axis = tilt_info["z_axis"]
    logger.info(
        f"[{label}] cls={cls} "
        f"p=[{p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f}]mm "
        f"x=[{x_axis[0]:+.3f}, {x_axis[1]:+.3f}, {x_axis[2]:+.3f}] "
        f"y=[{y_axis[0]:+.3f}, {y_axis[1]:+.3f}, {y_axis[2]:+.3f}] "
        f"z=[{z_axis[0]:+.3f}, {z_axis[1]:+.3f}, {z_axis[2]:+.3f}] "
        f"x_dot_up={tilt_info['x_dot_up']:+.3f} "
        f"y_dot_up={tilt_info['y_dot_up']:+.3f} "
        f"z_dot_up={tilt_info['z_dot_up']:+.3f} "
        f"x_tilt_ground={tilt_info['x_tilt_ground_deg']:.2f}deg "
        f"y_tilt_ground={tilt_info['y_tilt_ground_deg']:.2f}deg "
        f"z_tilt_up={tilt_info['z_tilt_up_deg']:.2f}deg"
    )


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

        # Empty-space candidates from mixed_pose_vision_node.
        # Used when tilted_grasp marks the output object id negative.
        # The transform node converts candidate/object camera translations to
        # base/world coordinates and chooses a safe empty XY.
        self.declare_parameter("empty_space_topic", "/empty_space_candidates")
        self.declare_parameter("empty_space_override_on_negative_tilted_id", True)
        self.declare_parameter("empty_space_candidate_max_age_sec", 5.0)
        # Full object footprint size. World filtering uses a circular exclusion
        # radius = hypot(size_x/2, size_y/2) + safety_margin.
        # For 60x60 mm object, base radius is about 42.4 mm.
        self.declare_parameter("empty_space_object_size_x_mm", 60.0)
        self.declare_parameter("empty_space_object_size_y_mm", 60.0)
        self.declare_parameter("empty_space_safety_margin_mm", 10.0)
        self.declare_parameter("empty_space_roi_edge_reject_px", 0.0)
        # Weighted selection after hard filtering.
        # clearance: far from occupied objects; pose6_proximity: close to original tilted pose6 XY.
        self.declare_parameter("empty_space_w_clearance", 2.0)
        self.declare_parameter("empty_space_w_pose6_proximity", 2.0)
        self.declare_parameter("empty_space_w_roi_edge", 1.0)
        self.declare_parameter("empty_space_w_vision_score", 0.5)
        # Saturation values for score normalization. Values above these become 1.0.
        self.declare_parameter("empty_space_clearance_norm_mm", 150.0)
        self.declare_parameter("empty_space_pose6_proximity_norm_mm", 200.0)
        self.declare_parameter("empty_space_roi_edge_norm_px", 120.0)

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
        self.empty_space_topic = str(self.get_parameter("empty_space_topic").value)

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

        self.latest_empty_space_payload = None
        self.latest_empty_space_stamp_sec = None

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
        self.empty_space_sub = self.create_subscription(
            String, self.empty_space_topic, self.empty_space_callback, 10
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
            "object output=/vision/peg_targets: success len=7 [selected_object_id, tcp_moveL_pose6] or len=9 [-id, original_pose6, empty_space_world_xy], failure len!=7 [visible_ids]; "
            "insert output=/vision/hole_targets: [x_mm, y_mm, yaw_deg, id]. "
            f"empty_space_topic={self.empty_space_topic}."
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
                # Optional. Used only for hole/cross by default in object_to_pose6_target_dict.
                "tilted_grasp": obj_data.get("tilted_grasp", {}) or {},
                # Optional. Used for cylinder yaw candidate scoring.
                "cylinder_yaw_search": obj_data.get("cylinder_yaw_search", {}) or {},
            }

        self.get_logger().info(f"loaded object target YAML: {path}")
        return cfg

    def get_object_target_transforms_for_class(
        self,
        cls: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
                "tilted_grasp": {},
                "cylinder_yaw_search": {},
            }

        raw_object_T_centered_object = np.asarray(
            item["raw_object_T_centered_object"], dtype=np.float64
        ).reshape(4, 4)
        centered_object_T_tcp_goal = np.asarray(
            item["centered_object_T_tcp_goal"], dtype=np.float64
        ).reshape(4, 4)
        symmetry = item.get("symmetry", {}) or {}
        tilted_grasp = item.get("tilted_grasp", {}) or {}
        cylinder_yaw_search = item.get("cylinder_yaw_search", {}) or {}

        validate_T(raw_object_T_centered_object, name=f"raw_object_T_centered_object[{cls}]")
        validate_T(centered_object_T_tcp_goal, name=f"centered_object_T_tcp_goal[{cls}]")

        return (
            raw_object_T_centered_object.copy(),
            centered_object_T_tcp_goal.copy(),
            dict(symmetry),
            dict(tilted_grasp),
            dict(cylinder_yaw_search),
        )

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

            # Preferred synchronized path:
            # mixed_pose_vision_node bundles empty-space candidates into the same
            # /object_poses JSON as the 6D pose result. This avoids cross-topic
            # callback ordering races between /empty_space_candidates and /object_poses.
            bundled_empty = data.get("empty_space", None)
            if isinstance(bundled_empty, dict) and str(bundled_empty.get("mode", "")) == "empty_space":
                self.latest_empty_space_payload = bundled_empty
                self.latest_empty_space_stamp_sec = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(
                    f"[EMPTY-SPACE] bundled with object JSON: "
                    f"candidates={len(bundled_empty.get('candidates', []))} "
                    f"objects_camera={len(bundled_empty.get('objects_camera', []))} "
                    f"trigger_seq={bundled_empty.get('trigger_seq', None)}"
                )
            else:
                # Do not reuse a stale empty-space payload from a previous trigger.
                # If the object JSON does not contain empty_space, the empty-space
                # override will be skipped and the original pose6 XY will be used.
                self.latest_empty_space_payload = None
                self.latest_empty_space_stamp_sec = None
        except Exception as e:
            self.latest_objects = []
            self.latest_object_available_ids = []
            self.latest_object_status = "parse_error"
            self.latest_empty_space_payload = None
            self.latest_empty_space_stamp_sec = None
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

    def empty_space_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("empty-space payload must be a JSON object")
            if str(data.get("mode", "")) != "empty_space":
                raise ValueError(f"unexpected empty-space mode={data.get('mode')}")

            # Debug-only topic path. The synchronized source of truth is the
            # empty_space field bundled inside /object_poses. To avoid stale/racy
            # cross-topic data, do not update latest_empty_space_payload here.
            self.get_logger().info(
                f"[EMPTY-SPACE] debug topic received candidates={len(data.get('candidates', []))} "
                f"objects_camera={len(data.get('objects_camera', []))} "
                f"trigger_seq={data.get('trigger_seq', None)} "
                f"use_for_control=False"
            )
        except Exception as e:
            self.latest_empty_space_payload = None
            self.latest_empty_space_stamp_sec = None
            self.get_logger().warn(f"failed to parse {self.empty_space_topic} JSON: {e}")

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
    # Empty-space world XY selection
    # ============================================================

    def camera_point_m_to_base_mm(self, camera_xyz_m: Dict[str, Any], base_T_ee: np.ndarray) -> np.ndarray:
        """Transform a camera-frame point in meters to base/world point in mm."""
        p_cam_mm = np.array([
            float(camera_xyz_m["x"]) * 1000.0,
            float(camera_xyz_m["y"]) * 1000.0,
            float(camera_xyz_m["z"]) * 1000.0,
            1.0,
        ], dtype=np.float64)

        base_T_cam = np.asarray(base_T_ee, dtype=np.float64).reshape(4, 4) @ self.ee_T_cam
        p_base_mm = base_T_cam @ p_cam_mm
        return np.asarray(p_base_mm[:3], dtype=np.float64).reshape(3)

    @staticmethod
    def roi_edge_distance_px(pixel: Dict[str, Any], roi: Dict[str, Any]) -> float:
        """Distance from a pixel candidate to the rectangular ROI boundary in pixels."""
        if not isinstance(pixel, dict) or not isinstance(roi, dict):
            return 0.0
        if roi.get("type") != "rect":
            return 0.0

        u = float(pixel.get("u", 0.0))
        v = float(pixel.get("v", 0.0))
        x_min = float(roi.get("x_min", 0.0))
        y_min = float(roi.get("y_min", 0.0))
        x_max = float(roi.get("x_max", x_min))
        y_max = float(roi.get("y_max", y_min))

        return float(min(u - x_min, x_max - u, v - y_min, y_max - v))

    def empty_payload_is_fresh(self) -> bool:
        if self.latest_empty_space_payload is None or self.latest_empty_space_stamp_sec is None:
            return False

        max_age = float(self.get_parameter("empty_space_candidate_max_age_sec").value)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        age_sec = float(now_sec - float(self.latest_empty_space_stamp_sec))
        return age_sec <= max_age

    def select_empty_space_world_xy(
        self,
        base_T_ee: np.ndarray,
        fallback_xy_mm: Tuple[float, float],
    ) -> Optional[Dict[str, Any]]:
        """Select a final empty-space world XY from the latest vision candidates.

        Policy:
          1) Convert candidate camera xyz and object camera translations to base/world mm.
          2) Hard reject candidates whose center is too close to any object center.
             The 60x60 mm footprint is approximated as a circle using the
             circumscribed radius: hypot(size_x/2, size_y/2) + safety_margin.
          3) Hard reject candidates too close to the ROI edge when configured.
          4) Rank remaining candidates by a weighted score that considers both:
                - object clearance: farther from objects is better.
                - pose6 proximity: closer to the original tilted pose6 XY is better.
             This treats object adjacency and excessive displacement from the
             original pose as two simultaneous "do not place here" costs.
          5) If nothing remains, return None and caller keeps the original pose6 XY.
        """
        if not self.empty_payload_is_fresh():
            return None

        payload = self.latest_empty_space_payload
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        objects_camera = payload.get("objects_camera", []) if isinstance(payload, dict) else []
        roi = payload.get("roi", {}) if isinstance(payload, dict) else {}

        if not isinstance(candidates, list) or len(candidates) == 0:
            return None

        object_size_x_mm = float(self.get_parameter("empty_space_object_size_x_mm").value)
        object_size_y_mm = float(self.get_parameter("empty_space_object_size_y_mm").value)
        safety_margin_mm = float(self.get_parameter("empty_space_safety_margin_mm").value)
        roi_edge_reject_px = float(self.get_parameter("empty_space_roi_edge_reject_px").value)

        w_clearance = float(self.get_parameter("empty_space_w_clearance").value)
        w_pose6 = float(self.get_parameter("empty_space_w_pose6_proximity").value)
        w_edge = float(self.get_parameter("empty_space_w_roi_edge").value)
        w_vision = float(self.get_parameter("empty_space_w_vision_score").value)

        clearance_norm_mm = max(float(self.get_parameter("empty_space_clearance_norm_mm").value), 1e-6)
        pose6_proximity_norm_mm = max(float(self.get_parameter("empty_space_pose6_proximity_norm_mm").value), 1e-6)
        roi_edge_norm_px = max(float(self.get_parameter("empty_space_roi_edge_norm_px").value), 1e-6)

        object_radius_mm = float(np.hypot(object_size_x_mm * 0.5, object_size_y_mm * 0.5))
        reject_radius_mm = object_radius_mm + safety_margin_mm

        fallback_xy = np.asarray(fallback_xy_mm, dtype=np.float64).reshape(2)

        # Transform detected object centers to base/world XY.
        object_world_rows = []
        for obj in objects_camera:
            try:
                cam = obj.get("camera_translation", None)
                if cam is None:
                    continue
                p_obj = self.camera_point_m_to_base_mm(cam, base_T_ee)
                object_world_rows.append({
                    "class_name": str(obj.get("class_name", "unknown")),
                    "class_id": int(obj.get("class_id", -1)),
                    "world": p_obj,
                })
            except Exception:
                continue

        valid_rows = []
        rejected_by_object = 0
        rejected_by_roi_edge = 0
        rejected_invalid = 0

        for cand in candidates:
            try:
                cam = cand.get("camera", None)
                pixel = cand.get("pixel", {})
                if cam is None:
                    rejected_invalid += 1
                    continue

                p_cand = self.camera_point_m_to_base_mm(cam, base_T_ee)
                edge_px = self.roi_edge_distance_px(pixel, roi)

                if edge_px < roi_edge_reject_px:
                    rejected_by_roi_edge += 1
                    continue

                min_center_dist_mm = float("inf")
                min_circle_clearance_mm = float("inf")
                nearest_obj_debug = None
                blocked = False

                for obj_row in object_world_rows:
                    p_obj = obj_row["world"]
                    dx = float(p_cand[0] - p_obj[0])
                    dy = float(p_cand[1] - p_obj[1])
                    center_dist = float(np.hypot(dx, dy))
                    circle_clearance = float(center_dist - reject_radius_mm)

                    if center_dist < min_center_dist_mm:
                        min_center_dist_mm = center_dist
                        nearest_obj_debug = obj_row
                    if circle_clearance < min_circle_clearance_mm:
                        min_circle_clearance_mm = circle_clearance

                    # 60x60 mm object footprint is treated as a rotation-invariant circle.
                    if center_dist < reject_radius_mm:
                        blocked = True
                        break

                if blocked:
                    rejected_by_object += 1
                    continue

                if not np.isfinite(min_center_dist_mm):
                    min_center_dist_mm = 999999.0
                if not np.isfinite(min_circle_clearance_mm):
                    min_circle_clearance_mm = 999999.0

                dist_to_fallback_xy_mm = float(
                    np.hypot(float(p_cand[0] - fallback_xy[0]), float(p_cand[1] - fallback_xy[1]))
                )
                vision_score = float(cand.get("score", 0.0))

                valid_rows.append({
                    "candidate": cand,
                    "world": p_cand,
                    "min_center_dist_mm": float(min_center_dist_mm),
                    "min_circle_clearance_mm": float(min_circle_clearance_mm),
                    "dist_to_fallback_xy_mm": float(dist_to_fallback_xy_mm),
                    "roi_edge_distance_px": float(edge_px),
                    "vision_score_raw": float(vision_score),
                    "nearest_object": None if nearest_obj_debug is None else {
                        "class_name": nearest_obj_debug["class_name"],
                        "class_id": nearest_obj_debug["class_id"],
                        "world_xy_mm": [
                            round(float(nearest_obj_debug["world"][0]), 3),
                            round(float(nearest_obj_debug["world"][1]), 3),
                        ],
                    },
                })

            except Exception:
                rejected_invalid += 1
                continue

        if not valid_rows:
            self.get_logger().warn(
                "[EMPTY-SPACE-WORLD] no valid final candidate after world filtering. "
                f"total={len(candidates)} rejected_object={rejected_by_object} "
                f"rejected_roi_edge={rejected_by_roi_edge} rejected_invalid={rejected_invalid}. "
                f"fallback_xy=({fallback_xy_mm[0]:.1f}, {fallback_xy_mm[1]:.1f})mm"
            )
            return None

        # Normalize vision scores by the maximum score in the current valid set.
        max_vision = max([max(0.0, float(r["vision_score_raw"])) for r in valid_rows] + [1e-6])

        for row in valid_rows:
            # Object clearance: larger is better. Saturate to prevent one very far point
            # from dominating pose6 proximity completely.
            norm_clearance = float(np.clip(row["min_circle_clearance_mm"] / clearance_norm_mm, 0.0, 1.0))

            # Pose6 proximity: closer to original tilted pose6 XY is better.
            # 0mm distance => 1.0, distance >= pose6_proximity_norm_mm => 0.0.
            norm_pose6_proximity = float(
                np.clip(1.0 - row["dist_to_fallback_xy_mm"] / pose6_proximity_norm_mm, 0.0, 1.0)
            )

            # ROI edge: farther from ROI boundary is better.
            norm_edge = float(np.clip(row["roi_edge_distance_px"] / roi_edge_norm_px, 0.0, 1.0))
            norm_vision = float(np.clip(max(0.0, row["vision_score_raw"]) / max_vision, 0.0, 1.0))

            score = (
                w_clearance * norm_clearance
                + w_pose6 * norm_pose6_proximity
                + w_edge * norm_edge
                + w_vision * norm_vision
            )

            row["score"] = float(score)
            row["score_terms"] = {
                "norm_clearance": float(norm_clearance),
                "norm_pose6_proximity": float(norm_pose6_proximity),
                "norm_roi_edge": float(norm_edge),
                "norm_vision": float(norm_vision),
                "weighted_clearance": float(w_clearance * norm_clearance),
                "weighted_pose6_proximity": float(w_pose6 * norm_pose6_proximity),
                "weighted_roi_edge": float(w_edge * norm_edge),
                "weighted_vision": float(w_vision * norm_vision),
            }
            row["priority_tuple"] = (
                float(score),
                float(row["min_circle_clearance_mm"]),
                -float(row["dist_to_fallback_xy_mm"]),
                float(row["roi_edge_distance_px"]),
                float(row["vision_score_raw"]),
            )

        valid_rows.sort(key=lambda row: row["priority_tuple"], reverse=True)
        selected = valid_rows[0]
        p = selected["world"]
        cand = selected["candidate"]

        result = {
            "selected_world_xy_mm": [float(p[0]), float(p[1])],
            "selected_world_xyz_mm": [float(p[0]), float(p[1]), float(p[2])],
            "selected_candidate_rank": int(cand.get("rank", -1)),
            "selected_candidate_pixel": cand.get("pixel", {}),
            "selected_candidate_camera": cand.get("camera", {}),
            "score": float(selected["score"]),
            "score_terms": selected["score_terms"],
            "min_center_dist_mm": float(selected["min_center_dist_mm"]),
            "min_circle_clearance_mm": float(selected["min_circle_clearance_mm"]),
            "dist_to_original_pose6_xy_mm": float(selected["dist_to_fallback_xy_mm"]),
            "roi_edge_distance_px": float(selected["roi_edge_distance_px"]),
            "nearest_object": selected["nearest_object"],
            "selection_policy": {
                "type": "hard_filter_then_weighted_score",
                "hard_filters": [
                    "center_dist_to_any_object >= reject_radius_mm",
                    "roi_edge_distance_px >= empty_space_roi_edge_reject_px",
                ],
                "score": "w_clearance*norm_clearance + w_pose6*norm_pose6_proximity + w_edge*norm_roi_edge + w_vision*norm_vision",
                "meaning": "far from objects and close to original pose6 XY are considered simultaneously",
                "weights": {
                    "w_clearance": float(w_clearance),
                    "w_pose6_proximity": float(w_pose6),
                    "w_roi_edge": float(w_edge),
                    "w_vision_score": float(w_vision),
                },
                "norms": {
                    "clearance_norm_mm": float(clearance_norm_mm),
                    "pose6_proximity_norm_mm": float(pose6_proximity_norm_mm),
                    "roi_edge_norm_px": float(roi_edge_norm_px),
                },
            },
            "world_filter_params": {
                "object_size_x_mm": object_size_x_mm,
                "object_size_y_mm": object_size_y_mm,
                "object_radius_mm": object_radius_mm,
                "safety_margin_mm": safety_margin_mm,
                "reject_radius_mm": reject_radius_mm,
                "roi_edge_reject_px": roi_edge_reject_px,
            },
            "stats": {
                "input_candidates": int(len(candidates)),
                "valid_after_world_filter": int(len(valid_rows)),
                "rejected_by_object_size": int(rejected_by_object),
                "rejected_by_roi_edge": int(rejected_by_roi_edge),
                "rejected_invalid": int(rejected_invalid),
                "objects_world": int(len(object_world_rows)),
            },
        }

        self.get_logger().info(
            "[EMPTY-SPACE-WORLD] selected "
            f"rank={result['selected_candidate_rank']} "
            f"pixel={result['selected_candidate_pixel']} "
            f"world_xy=({p[0]:.1f}, {p[1]:.1f})mm "
            f"score={result['score']:.3f} "
            f"circle_clearance={result['min_circle_clearance_mm']:.1f}mm "
            f"pose6_dist={result['dist_to_original_pose6_xy_mm']:.1f}mm "
            f"roi_edge={result['roi_edge_distance_px']:.1f}px "
            f"valid={len(valid_rows)}/{len(candidates)}"
        )

        return result

    def apply_empty_space_override_if_needed(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """For tilted_grasp negative output id, append selected empty-space world X/Y.

        Output policy:
          - Do NOT replace the original object grasp pose6.
          - Keep target_pose6 as the pose computed from the object 6D pose.
          - Normal object:
                [id, pose6]                                      len=7
          - Tilted object + valid empty-space:
                [-id, pickup_pose6, empty_x, empty_y]             len=9
          - Tilted object + no valid empty-space:
                [-id, pickup_pose6, pickup_x, pickup_y, -99]      len=10

        Meaning:
          - data[1:7] is always the pickup pose6.
          - data[7], data[8] are empty-space world XY if valid.
          - if data[9] == -99, empty-space is invalid and data[7:9] is fallback pickup XY.
        """
        out = dict(target)
        target_pose6 = np.asarray(target["target_pose6"], dtype=np.float64).reshape(6).copy()

        target_id = int(round(float(target.get("id", 999999))))
        tilted_grasp_used = bool(target.get("tilted_grasp_used", False))
        enable_append = bool(self.get_parameter("empty_space_override_on_negative_tilted_id").value)
        should_append = bool(enable_append and tilted_grasp_used and target_id < 0)

        # Always keep original pickup pose6.
        # Empty-space XY is metadata appended after pose6, not replacement of pose6 x/y.
        out["target_pose6"] = target_pose6

        # Reset empty-space fields first.
        out["empty_space_override_checked"] = bool(tilted_grasp_used and target_id < 0)
        out["empty_space_override_used"] = False
        out["empty_space_override_reason"] = "not_tilted_grasp_negative_output_id"

        out["empty_space_append_checked"] = bool(tilted_grasp_used and target_id < 0)
        out["empty_space_append_used"] = False
        out["empty_space_append_reason"] = "not_tilted_grasp_negative_output_id"

        out.pop("empty_space_world_xy_mm", None)
        out.pop("empty_space_selected", None)
        out.pop("empty_space_status_code", None)

        if not should_append:
            if not enable_append:
                out["empty_space_override_reason"] = "disabled_by_parameter"
                out["empty_space_append_reason"] = "disabled_by_parameter"
            return out

        base_T_ee = pose6_mm_deg_to_T_mm(self.pending_trigger_msg.data[3:9])
        validate_T(base_T_ee, name="empty_space_append_base_T_ee")

        # fallback XY is original pickup pose6 x/y in base/world frame.
        fallback_xy = (float(target_pose6[0]), float(target_pose6[1]))

        selected = self.select_empty_space_world_xy(
            base_T_ee,
            fallback_xy_mm=fallback_xy,
        )

        if selected is None:
            # No valid empty-space candidate.
            # Still publish a tilted-success response with fallback pickup XY and status -99:
            #   [-id, pickup_pose6, pickup_x, pickup_y, -99]
            # This prevents the controller from mistaking the response for a normal len=7 success.
            out["empty_space_world_xy_mm"] = [
                float(fallback_xy[0]),
                float(fallback_xy[1]),
            ]
            out["empty_space_selected"] = None
            out["empty_space_status_code"] = -99.0

            out["empty_space_override_used"] = False
            out["empty_space_override_reason"] = (
                "no_valid_empty_space_candidate_fallback_to_original_pose6_xy"
            )

            out["empty_space_append_used"] = False
            out["empty_space_append_reason"] = (
                "no_valid_empty_space_candidate_append_original_pose6_xy_with_minus99"
            )

            self.get_logger().warn(
                "[EMPTY-SPACE-WORLD] no valid empty-space candidate. "
                f"append fallback pickup_xy=({fallback_xy[0]:.1f}, {fallback_xy[1]:.1f})mm "
                "status=-99"
            )

            return out

        x_mm, y_mm = selected["selected_world_xy_mm"]

        # Valid empty-space case:
        #   [-id, pickup_pose6, empty_x, empty_y]
        out["empty_space_world_xy_mm"] = [float(x_mm), float(y_mm)]
        out["empty_space_selected"] = selected
        out.pop("empty_space_status_code", None)

        out["empty_space_override_used"] = False
        out["empty_space_override_reason"] = "append_only_original_pose6_kept"

        out["empty_space_append_used"] = True
        out["empty_space_append_reason"] = (
            "tilted_grasp_negative_output_id_append_empty_space_xy"
        )

        return out

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
            raw_object_T_centered_object, centered_object_T_tcp_nominal, symmetry, tilted_grasp_cfg, cylinder_yaw_cfg = (
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
            # 그 후, hole/cross에 한해서 selected +X가 너무 기울어져 있으면
            # centered object local +Z 기준 45deg yaw를 먼저 적용하고 다시 XY_FLAT을 수행한다.
            xy_info = None
            tilted_grasp_used = False
            tilted_grasp_reason = ""
            tilted_pre_yaw_deg = 0.0
            tilt_info = compute_centered_axis_tilt_info(base_T_centered_object)

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

                # Debug: centered object frame after:
                # raw object -> object_to_center -> z-up defense -> xy-flat canonicalization
                # This is BEFORE centered_object_to_tcp.
                tilt_info = compute_centered_axis_tilt_info(base_T_centered_object)
                log_centered_axis_tilt(
                    self.get_logger(),
                    "CENTERED_OBJECT_SAFE",
                    cls,
                    tilt_info,
                )

                # cylinder는 제외. hole/cross만 tilted grasp branch 허용.
                tilted_enable = (
                    bool(tilted_grasp_cfg.get("enable", False))
                    and canonical_object_name(cls) != "cylinder"
                )
                tilted_threshold_deg = float(tilted_grasp_cfg.get("x_tilt_threshold_deg", 999.0))
                tilted_pre_yaw_deg = float(tilted_grasp_cfg.get("pre_yaw_deg", 45.0))

                if tilted_enable and tilt_info["x_tilt_ground_deg"] >= tilted_threshold_deg:
                    before_x_tilt = float(tilt_info["x_tilt_ground_deg"])

                    # centered object local +Z 기준으로 먼저 45deg yaw 회전.
                    # base_T_centered_object의 오른쪽에 곱하므로 local frame 기준 yaw가 된다.
                    base_T_centered_object = base_T_centered_object @ T_rot_z_deg(tilted_pre_yaw_deg)
                    validate_T(
                        base_T_centered_object,
                        name=f"base_T_centered_object_tilted_yaw[{cls}]",
                    )

                    # 45deg 돌린 뒤, 다시 x/y 중 더 world XY 평면에 완만한 축을 +X로 선택.
                    base_T_centered_object, xy_info_2 = canonicalize_xy_flatter_as_x(
                        base_T_centered_object,
                        swap_margin=float(self.get_parameter("canonicalize_xy_swap_margin").value),
                    )

                    tilt_info_2 = compute_centered_axis_tilt_info(base_T_centered_object)

                    tilted_grasp_used = True
                    tilted_grasp_reason = (
                        f"x_tilt_ground {before_x_tilt:.2f}deg >= "
                        f"threshold {tilted_threshold_deg:.2f}deg"
                    )

                    self.get_logger().warn(
                        f"[TILTED_GRASP_X45] cls={cls} "
                        f"reason='{tilted_grasp_reason}' "
                        f"pre_yaw={tilted_pre_yaw_deg:.1f}deg "
                        f"xy2_swapped={xy_info_2['xy_swapped']} "
                        f"xy2_selected_x={xy_info_2['selected_x_from']} "
                        f"x_tilt_before={before_x_tilt:.2f}deg "
                        f"x_tilt_after={tilt_info_2['x_tilt_ground_deg']:.2f}deg "
                        f"y_tilt_after={tilt_info_2['y_tilt_ground_deg']:.2f}deg "
                        f"z_tilt_up_after={tilt_info_2['z_tilt_up_deg']:.2f}deg"
                    )
                    log_centered_axis_tilt(
                        self.get_logger(),
                        "CENTERED_OBJECT_TILTED_SAFE",
                        cls,
                        tilt_info_2,
                    )

                    tilt_info = tilt_info_2
                    xy_info = {
                        **xy_info,
                        "tilted_xy_swapped": bool(xy_info_2["xy_swapped"]),
                        "tilted_selected_x_from": str(xy_info_2["selected_x_from"]),
                        "tilted_x_flatness_after": float(xy_info_2["x_flatness_after"]),
                        "tilted_y_flatness_after": float(xy_info_2["y_flatness_after"]),
                    }

                max_flat = float(self.get_parameter("canonicalize_xy_max_flatness").value)
                final_x_flatness = abs(float(tilt_info["x_dot_up"]))
                if final_x_flatness > max_flat:
                    self.get_logger().warn(
                        f"[SKIP] cls={cls}: selected object +X is still too steep. "
                        f"x_flatness_after={final_x_flatness:.3f} > {max_flat:.3f}"
                    )
                    return None

            # symmetry.yaw_candidates_deg 후보를 만들고, RB5 마지막 joint 기준으로 필터링.
            grasp_candidates = self.make_symmetric_grasp_candidates(
                centered_object_T_tcp_nominal,
                symmetry,
            )

            best = None
            is_cylinder = canonical_object_name(cls) == "cylinder"
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # Cylinder-specific policy:
            #   1) J5 limit 초과 후보는 기존처럼 skip
            #   2) 남은 후보 중 centered object 기준 축이 world XY 평면에 가장 평평한 후보 선택
            #   3) 평평함이 비슷하면 J5 delta가 더 작은 후보 선택
            #
            # Default score_axis="object_x" means:
            #   yaw 후보가 적용된 centered object frame의 +X축을 평가한다.
            #   즉 TCP 축이 아니라 물체/centered-object 축 기준이다.
            cyl_yaw_search_enabled = bool(cylinder_yaw_cfg.get("enable", True))
            cyl_score_axis = str(cylinder_yaw_cfg.get("score_axis", "object_x")).lower()
            cyl_flat_tolerance_deg = float(cylinder_yaw_cfg.get("flat_tolerance_deg", 3.0))

            def _axis_for_cylinder_score(
                axis_name: str,
                base_T_centered: np.ndarray,
                base_T_tcp: np.ndarray,
                yaw_deg: float,
            ) -> np.ndarray:
                axis_name = str(axis_name).lower()

                object_axis_map = {
                    "object_x": 0, "+object_x": 0, "center_x": 0, "+center_x": 0,
                    "centered_object_x": 0, "+centered_object_x": 0,
                    "object_y": 1, "+object_y": 1, "center_y": 1, "+center_y": 1,
                    "centered_object_y": 1, "+centered_object_y": 1,
                    "object_z": 2, "+object_z": 2, "center_z": 2, "+center_z": 2,
                    "centered_object_z": 2, "+centered_object_z": 2,
                }
                tcp_axis_map = {
                    "tcp_x": 0, "+tcp_x": 0,
                    "tcp_y": 1, "+tcp_y": 1,
                    "tcp_z": 2, "+tcp_z": 2,
                }

                if axis_name in object_axis_map:
                    # 후보 yaw가 적용된 centered object frame의 축을 평가.
                    # 여기서 오른쪽에 Rz(yaw)를 곱하므로 local +Z 기준 yaw가 된다.
                    base_T_centered_yaw = base_T_centered @ T_rot_z_deg(float(yaw_deg))
                    R_eval = orthonormalize_R(base_T_centered_yaw[:3, :3])
                    return R_eval[:, object_axis_map[axis_name]]

                if axis_name in tcp_axis_map:
                    # 필요 시 디버그/실험용: 최종 TCP frame의 축을 평가.
                    R_eval = orthonormalize_R(base_T_tcp[:3, :3])
                    return R_eval[:, tcp_axis_map[axis_name]]

                # 잘못된 값이면 안전하게 object_x 기준으로 fallback.
                base_T_centered_yaw = base_T_centered @ T_rot_z_deg(float(yaw_deg))
                R_eval = orthonormalize_R(base_T_centered_yaw[:3, :3])
                return R_eval[:, 0]

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

                axis_tilt_ground_deg = None
                score = delta_j5
                better = False

                if is_cylinder and cyl_yaw_search_enabled:
                    flat_axis = _axis_for_cylinder_score(
                        cyl_score_axis,
                        base_T_centered_object,
                        base_T_tcp_goal_cand,
                        cand["yaw_deg"],
                    )
                    axis_dot_up = abs(float(np.dot(flat_axis, world_up)))
                    axis_tilt_ground_deg = float(
                        np.degrees(np.arcsin(np.clip(axis_dot_up, 0.0, 1.0)))
                    )

                    if best is None:
                        better = True
                    else:
                        best_axis_tilt = float(best.get("axis_tilt_ground_deg", 999.0))
                        if axis_tilt_ground_deg < best_axis_tilt - cyl_flat_tolerance_deg:
                            better = True
                        elif abs(axis_tilt_ground_deg - best_axis_tilt) <= cyl_flat_tolerance_deg:
                            better = delta_j5 < float(best["delta_j5"])
                else:
                    # Non-cylinder, or cylinder_yaw_search.enable=false:
                    # J5 limit을 통과한 후보 중 J5 delta가 가장 작은 후보 선택.
                    if best is None or score < best["score"]:
                        better = True

                if better:
                    best = {
                        "candidate": cand,
                        "base_T_tcp_goal": base_T_tcp_goal_cand,
                        "estimated_j5": estimated_j5,
                        "delta_j5": delta_j5,
                        "signed_delta_j5": signed_delta_j5,
                        "score": score,
                        "axis_tilt_ground_deg": axis_tilt_ground_deg,
                        "cylinder_score_axis": cyl_score_axis if is_cylinder else "",
                        "cylinder_flat_tolerance_deg": cyl_flat_tolerance_deg if is_cylinder else 0.0,
                    }

                if is_cylinder:
                    if cyl_yaw_search_enabled:
                        self.get_logger().info(
                            f"[J5_CAND] cls={cls} cand={cand['name']} "
                            f"yaw={cand['yaw_deg']:.1f} "
                            f"est_J5={estimated_j5:.2f} "
                            f"signed_delta={signed_delta_j5:.2f} "
                            f"delta={delta_j5:.2f} "
                            f"axis={cyl_score_axis} "
                            f"axis_tilt_ground={axis_tilt_ground_deg:.2f}deg "
                            f"flat_tol={cyl_flat_tolerance_deg:.2f}deg"
                        )
                    else:
                        self.get_logger().info(
                            f"[J5_CAND] cls={cls} cand={cand['name']} "
                            f"yaw={cand['yaw_deg']:.1f} "
                            f"est_J5={estimated_j5:.2f} "
                            f"signed_delta={signed_delta_j5:.2f} "
                            f"delta={delta_j5:.2f} "
                            f"score={score:.2f} "
                            f"cylinder_yaw_search=disabled"
                        )
                else:
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

            output_obj_id = obj_id
            if tilted_grasp_used and bool(tilted_grasp_cfg.get("mark_output_id_negative", True)):
                # cylinder는 tilted_grasp branch에서 제외되므로 -0 문제는 발생하지 않는다.
                output_obj_id = -abs(obj_id)

            return {
                "class": cls,
                "id": output_obj_id,
                "raw_id": obj_id,
                "tilted_grasp_used": bool(tilted_grasp_used),
                "tilted_grasp_reason": tilted_grasp_reason,
                "tilted_pre_yaw_deg": float(tilted_pre_yaw_deg),
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
                "selected_axis_tilt_ground_deg": (
                    None if best.get("axis_tilt_ground_deg") is None
                    else float(best["axis_tilt_ground_deg"])
                ),
                "cylinder_score_axis": str(best.get("cylinder_score_axis", "")),
                "cylinder_flat_tolerance_deg": float(best.get("cylinder_flat_tolerance_deg", 0.0)),
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

            normal object:
                [object_id,
                 tcp_x_mm, tcp_y_mm, tcp_z_mm, tcp_rx_deg, tcp_ry_deg, tcp_rz_deg]
                len = 7

            tilted object with valid empty-space:
                [-object_id,
                 pickup_x_mm, pickup_y_mm, pickup_z_mm,
                 pickup_rx_deg, pickup_ry_deg, pickup_rz_deg,
                 empty_space_world_x_mm, empty_space_world_y_mm]
                len = 9

            tilted object without valid empty-space:
                [-object_id,
                 pickup_x_mm, pickup_y_mm, pickup_z_mm,
                 pickup_rx_deg, pickup_ry_deg, pickup_rz_deg,
                 pickup_x_mm, pickup_y_mm,
                 -99]
                len = 10

        Important:
            - For negative id, pose6 is still the original object pickup/grasp pose6.
            - Empty-space XY is appended after pose6.
            - pose6 X/Y is never replaced.
            - status -99 means empty-space candidate is invalid/missing.
        """
        data = []

        for t in targets:
            obj_id = float(t["id"])
            pose6 = np.asarray(t["target_pose6"], dtype=np.float64).reshape(6)

            row = [obj_id] + pose6.tolist()

            extra_xy = t.get("empty_space_world_xy_mm", None)
            if int(round(obj_id)) < 0 and extra_xy is not None:
                extra_xy = np.asarray(extra_xy, dtype=np.float64).reshape(2)

                row.extend([
                    float(extra_xy[0]),
                    float(extra_xy[1]),
                ])

                # Only fallback/no-empty-space case has empty_space_status_code.
                # Valid empty-space case keeps len=9.
                status_code = t.get("empty_space_status_code", None)
                if status_code is not None:
                    row.append(float(status_code))

            data.extend(row)

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
        # If tilted_grasp marks the output id negative, keep original pose6 and
        # append selected empty-space world X/Y at the end of the message.
        target = self.apply_empty_space_override_if_needed(target)

        msg = Float64MultiArray()
        msg.data = self.object_targets_to_msg_data([target])

        debug_target = {
            "class": target["class"],
            "id": target["id"],
            "raw_id": target.get("raw_id", target["id"]),
            "tilted_grasp_used": target.get("tilted_grasp_used", False),
            "tilted_pre_yaw_deg": target.get("tilted_pre_yaw_deg", 0.0),
            "tilted_grasp_reason": target.get("tilted_grasp_reason", ""),
            "confidence": round(float(target["confidence"]), 3),
            "source_depth_m": target.get("source_depth_m", None),
            "target_frame": target["target_frame"],
            "selected_grasp": target.get("selected_grasp_name", None),
            "selected_yaw_deg": target.get("selected_grasp_yaw_deg", None),
            "estimated_last_joint_deg": target.get("estimated_last_joint_deg", None),
            "last_joint_delta_deg": target.get("last_joint_delta_deg", None),
            "target_pose6": np.asarray(target["target_pose6"]).round(3).tolist(),
            "target_T": np.asarray(target["target_T"]).round(3).tolist(),
            "empty_space_override_checked": target.get("empty_space_override_checked", False),
            "empty_space_override_used": target.get("empty_space_override_used", False),
            "empty_space_override_reason": target.get("empty_space_override_reason", ""),
            "empty_space_append_checked": target.get("empty_space_append_checked", False),
            "empty_space_append_used": target.get("empty_space_append_used", False),
            "empty_space_append_reason": target.get("empty_space_append_reason", ""),
            "empty_space_world_xy_mm": target.get("empty_space_world_xy_mm", None),
            "empty_space_selected": target.get("empty_space_selected", None),
        }

        self.get_logger().info(
            f"[PUBLISH] topic={topic_name} type={label} format=id_plus_moveL_pose6_7_or_negative_id_pose6_emptyxy_9_success "
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