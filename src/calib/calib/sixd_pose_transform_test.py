"""
Test node: /object_poses 가 오면 즉시 로봇 TCP를 직접 쿼리하고
최종 grasp pose6를 연산한 뒤 matplotlib 3D로 시각화한다.

- 트리거 토픽 없음 (peg_trigger 제거)
- 로봇 연동: rbpodo CobotData.request_data() 로 현재 TCP 직접 쿼리
- 실제 publish 없음 (viz only)
- USE_ROBOT=False 이면 더미 TCP 포즈로 대체

ROS params:
  robot_ip            str   "192.168.1.10"
  use_robot           bool  True
  object_topic        str   "/object_poses"
  handeye_result_path str   ""          # 비우면 calib 패키지 기본값
  object_grasp_yaml_path str ""
  min_confidence      float 0.3
  canonicalize_object_axes bool True
  canonicalize_z_flip_margin float 0.05
  visualize_axes_length_mm   float 50.0
  visualize_approach_length_mm float 80.0
  visualize_save_dir  str   ""
  
  
# 2. 빌드
cd ~/course/robot_manipulation-bin-picking
rm -rf build/calib install/calib
source /opt/ros/humble/setup.bash
conda activate cource
python -m colcon build --packages-select calib
source install/setup.bash

# 3. 실행
ros2 run calib sixd_pose_transform_test \
  --ros-args \
  -p robot_ip:=192.168.1.10 \
  -p use_robot:=true \
  -p object_topic:=/object_poses \
  -p handeye_result_path:=/home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/handeye_capture_rs/handeye_result.json \
  -p object_grasp_yaml_path:=/home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml

run:
  ros2 run <pkg> sixd_pose_transform_test \
    --ros-args \
    -p robot_ip:=192.168.1.10 \
    -p use_robot:=true \
    -p object_topic:=/object_poses \
    -p handeye_result_path:=/path/to/handeye_result.json \
    -p object_grasp_yaml_path:=/path/to/grasp.yaml
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


# ============================================================
# Rotation / Transform utils (동일 버전)
# ============================================================

def euler_zyx_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


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


def quat_xyzw_to_R(q_xyzw: List[float]) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return orthonormalize_R(R)


def orthonormalize_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt


def validate_T(T: np.ndarray, name: str = "T", atol: float = 1e-3):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name}: last row must be [0,0,0,1], got {T[3]}")
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(f"{name}: det(R)={det:.6f}")
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name}: R not orthogonal")


def pose6_mm_deg_to_T_mm(pose6) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    x, y, z, rx, ry, rz = pose6[:6]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def T_mm_to_pose6_mm_deg(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    validate_T(T, name="T_mm_to_pose6")
    R = orthonormalize_R(T[:3, :3])
    rpy = R_to_euler_zyx_deg(R)
    p = T[:3, 3]
    return np.array([p[0], p[1], p[2], rpy[0], rpy[1], rpy[2]], dtype=np.float64)


def T_m_to_T_mm(T_m: np.ndarray) -> np.ndarray:
    T = np.asarray(T_m, dtype=np.float64).reshape(4, 4).copy()
    T[:3, 3] *= 1000.0
    return T


def matrix_from_data(data, unit: str = "mm") -> np.ndarray:
    T = np.asarray(data, dtype=np.float64).reshape(4, 4).copy()
    T[:3, :3] = orthonormalize_R(T[:3, :3])
    if unit == "m":
        T[:3, 3] *= 1000.0
    validate_T(T, name="matrix_from_data")
    return T


def normalize_vec(v: np.ndarray, name: str = "v") -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"{name}: norm too small")
    return v / n


def defend_centered_object_z_up(
    T: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
    z_flip_margin: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    object +Z가 world +Z에 더 가깝도록 보정한다.

    방식:
      1. 원본 keep 후보 생성
      2. X/Z를 동시에 뒤집은 flip 후보 생성
      3. dot(object +Z, world +Z)가 더 큰 후보 선택
      4. 단, flip 후보가 margin 이상 더 좋을 때만 flip
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

    # 1. 원본 후보
    R_keep = _make_right_handed_from_xz(
        R[:, 0],
        R[:, 2],
        tag="keep",
    )
    keep_dot = float(np.dot(R_keep[:, 2], up))

    # 2. flip 후보
    # Z만 뒤집으면 det가 깨질 수 있으므로 X/Z를 같이 뒤집음
    R_flip = _make_right_handed_from_xz(
        -R[:, 0],
        -R[:, 2],
        tag="flip",
    )
    flip_dot = float(np.dot(R_flip[:, 2], up))

    # 3. flip이 충분히 더 위를 보면 flip
    z_flipped = flip_dot > keep_dot + float(z_flip_margin)

    if z_flipped:
        T[:3, :3] = R_flip
    else:
        T[:3, :3] = R_keep

    validate_T(T, name="z_defense_out")

    info = {
        "z_flipped": bool(z_flipped),

        # 기존 로그 호환용
        "before_dot_up": {
            "x": float(np.dot(R_keep[:, 0], up)),
            "y": float(np.dot(R_keep[:, 1], up)),
            "z": float(keep_dot),
        },

        # 추가 디버깅용
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



def canonicalize_xy_flatter_as_x(
    T: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
    swap_margin: float = 0.03,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    center/object frame의 X/Y 축 중 월드 XY 평면에 더 평행한 축을 새 object +X로 선택한다.

    목적:
      - 패러럴 그리퍼 닫힘축을 object X 계열에 맞출 때,
        object X가 바닥 쪽으로 가파르게 박히는 것을 줄인다.
      - 90도 대칭 물체에서 X/Y는 교환 가능하므로, 더 완만한 축을 canonical X로 둔다.

    기준:
      flatness = abs(dot(axis, world_up))
      flatness가 작을수록 월드 XY 평면에 더 평행함.
        0.0 → 완전 수평
        1.0 → 완전 수직

    동작:
      - abs(y_z) + margin < abs(x_z) 이면 X/Y를 90도 회전 교환
      - 아니면 원본 유지
    """
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

def object_json_to_cam_T_obj_mm(obj: Dict[str, Any]) -> np.ndarray:
    if "pose_matrix" in obj:
        T = np.asarray(obj["pose_matrix"], dtype=np.float64).reshape(4, 4)
        T[:3, :3] = orthonormalize_R(T[:3, :3])
        validate_T(T, name="pose_matrix")
        return T_m_to_T_mm(T)
    pos = obj["position"]
    ori = obj["orientation"]
    p = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=np.float64) * 1000.0
    if all(k in ori for k in ("x", "y", "z", "w")):
        R = quat_xyzw_to_R([ori["x"], ori["y"], ori["z"], ori["w"]])
    else:
        R = orthonormalize_R(np.column_stack([
            np.asarray(ori["axis_x"], dtype=np.float64),
            np.asarray(ori["axis_y"], dtype=np.float64),
            np.asarray(ori["axis_z"], dtype=np.float64),
        ]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    validate_T(T, name="cam_T_obj")
    return T


def canonical_object_name(cls: str) -> str:
    return {"cross_insert": "cross", "cylinder_insert": "cylinder", "hole_insert": "hole"}.get(cls, cls)


def wrap_deg(a: float) -> float:
    """각도를 [-180, 180) 범위로 정규화."""
    return float((float(a) + 180.0) % 360.0 - 180.0)


def T_rot_z_deg(deg: float) -> np.ndarray:
    """object/center local +Z 기준 yaw 회전 4x4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(0.0, 0.0, float(deg))
    return T


def trans_mm(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [float(x), float(y), float(z)]
    return T


def rot_y_deg(deg: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(0.0, float(deg), 0.0)
    return T


def rot_z_deg(deg: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(0.0, 0.0, float(deg))
    return T


def rb5_urdf_fk_T_tcp_mm(q_deg: np.ndarray) -> np.ndarray:
    """
    RB5 공식 URDF/joint.yaml 기준 간단 FK.
    q_deg = [J0, J1, J2, J3, J4, J5] in deg.
    반환 translation 단위는 mm.

    chain:
      Tz(169.2) Rz(q0)
      Ry(q1)
      Tz(425.0) Ry(q2)
      Tz(392.0) Ry(q3)
      T(0,-110.7,110.7) Rz(q4)
      Ry(q5)
      T(0,-96.7,0)
    """
    q = np.asarray(q_deg, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T = T @ trans_mm(0.0, 0.0, 169.2) @ rot_z_deg(q[0])
    T = T @ trans_mm(0.0, 0.0,   0.0) @ rot_y_deg(q[1])
    T = T @ trans_mm(0.0, 0.0, 425.0) @ rot_y_deg(q[2])
    T = T @ trans_mm(0.0, 0.0, 392.0) @ rot_y_deg(q[3])
    T = T @ trans_mm(0.0, -110.7, 110.7) @ rot_z_deg(q[4])
    T = T @ trans_mm(0.0, 0.0,   0.0) @ rot_y_deg(q[5])
    T = T @ trans_mm(0.0, -96.7, 0.0)
    T[:3, :3] = orthonormalize_R(T[:3, :3])
    validate_T(T, name="rb5_urdf_fk_T_tcp_mm")
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

def get_current_tcp_pose_vec_mm(robot_data) -> Optional[np.ndarray]:
    """
    rbpodo CobotData.request_data() 로 현재 TCP pose6 (mm, deg) 반환.
    실패 시 None.
    """
    try:
        state = robot_data.request_data()
        if state is None:
            return None
        # 필드명은 펌웨어 버전에 따라 다를 수 있음 (handeye_sampler 동일 패턴)
        sdata = state.sdata
        if hasattr(sdata, "tcp"):
            arr = np.array(sdata.tcp, dtype=np.float64)
        elif hasattr(sdata, "tcp_pos"):
            arr = np.array(sdata.tcp_pos, dtype=np.float64)
        elif hasattr(sdata, "cur_pos"):
            arr = np.array(sdata.cur_pos, dtype=np.float64)
        else:
            return None
        return arr[:6].copy()
    except Exception as e:
        return None


# ============================================================
# Visualization
# ============================================================

def set_axes_equal_3d(ax):
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    r = 0.5 * max(abs(xl[1]-xl[0]), abs(yl[1]-yl[0]), abs(zl[1]-zl[0]), 1.0)
    mx, my, mz = np.mean(xl), np.mean(yl), np.mean(zl)
    ax.set_xlim3d([mx-r, mx+r])
    ax.set_ylim3d([my-r, my+r])
    ax.set_zlim3d([mz-r, mz+r])


def draw_frame_3d(ax, T: np.ndarray, name: str, axis_len: float, alpha: float = 1.0):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = T[:3, 3]
    R = orthonormalize_R(T[:3, :3])
    for i, (color, lbl) in enumerate(zip(["r", "g", "b"], ["+X", "+Y", "+Z"])):
        v = R[:, i] * axis_len
        ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2],
                  color=color, alpha=alpha, arrow_length_ratio=0.18, linewidth=1.6)
    ax.text(p[0], p[1], p[2], f" {name}", fontsize=8)


def visualize_grasp_target(
    target: Dict[str, Any],
    current_tcp_pose6: np.ndarray,
    axis_len: float = 50.0,
    approach_len: float = 80.0,
    save_dir: str = "",
    fig_store: dict = None,
):
    """
    직관적 시각화 (필요한 것만):
      [cur TCP]  파란 다이아몬드 + RGB 축  → 지금 로봇이 어디 있는지
      [object]   회색 구                   → FP가 인식한 물체 위치
      [TCP goal] 빨간 별 + RGB 축 (굵게)  → 최종 그랩 목표 자세
      [approach] 마젠타 굵은 화살표        → TCP -Y 접근 방향 (아래를 향해야 정상)
      [pregrasp] 하늘색 X                  → approach 시작점
      [move]     초록 점선                 → 현재 TCP → goal 이동 경로
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[VIZ] matplotlib 없음, 시각화 스킵")
        return

    T_goal = np.asarray(target["target_T"],      dtype=np.float64).reshape(4, 4)
    T_raw  = np.asarray(target["base_T_obj_raw"], dtype=np.float64).reshape(4, 4)
    pose6  = np.asarray(target["target_pose6"],   dtype=np.float64).reshape(6)
    T_cur  = pose6_mm_deg_to_T_mm(current_tcp_pose6)

    p_goal = T_goal[:3, 3]
    p_obj  = T_raw[:3, 3]
    p_cur  = T_cur[:3, 3]

    R_goal       = orthonormalize_R(T_goal[:3, :3])
    approach_dir = -R_goal[:, 1]                        # TCP -Y
    p_pre        = p_goal + R_goal[:, 1] * approach_len # pregrasp = +Y 방향

    # ── 메타 정보 ──────────────────────────────
    cls       = target.get("class", "?")
    conf      = target.get("confidence", 0.0)
    zi        = target.get("axis_info") or {}
    z_flipped = zi.get("z_flipped", False)

    # approach 방향이 base -Z를 향하는지 간단 체크
    dot_z = float(np.dot(approach_dir, np.array([0, 0, 1])))
    approach_ok = dot_z < -0.5   # -Z 방향이면 True

    # ── figure 초기화 ──────────────────────────
    plt.ion()
    if fig_store is None:
        fig_store = {}
    if "fig" not in fig_store or fig_store["fig"] is None:
        fig_store["fig"] = plt.figure("Grasp Target Preview", figsize=(9, 7))
        fig_store["ax"]  = fig_store["fig"].add_subplot(111, projection="3d")
    else:
        fig_store["ax"].clear()

    ax = fig_store["ax"]

    # ── 타이틀 ────────────────────────────────
    approach_str = f"approach dot(-Z)={dot_z:.2f}  {'✓ OK' if approach_ok else '✗ CHECK'}"
    zflip_str    = f"z_defense={'ON' if z_flipped else 'off'}"
    ax.set_title(
        f"class={cls}  conf={conf:.3f}  {zflip_str}\n"
        f"goal  [{' '.join(f'{v:.1f}' for v in pose6)}]\n"
        f"{approach_str}",
        fontsize=9,
        color="red" if not approach_ok else "black",
    )
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")

    # ── 1. 현재 TCP (파란 계열, 반투명) ────────
    draw_frame_3d(ax, T_cur,  "cur TCP", axis_len * 0.8, alpha=0.4)
    ax.scatter(*p_cur, marker="D", s=80, color="steelblue",
               label="cur TCP", zorder=5)

    # ── 2. 물체 위치 (회색 구) ─────────────────
    ax.scatter(*p_obj, marker="o", s=120, color="dimgray",
               label=f"object ({cls})", zorder=4)
    ax.text(p_obj[0], p_obj[1], p_obj[2] + axis_len * 0.3,
            f" {cls}", fontsize=8, color="dimgray")

    # ── 3. TCP goal 프레임 (굵게, 불투명) ──────
    draw_frame_3d(ax, T_goal, "TCP goal", axis_len * 1.2, alpha=1.0)
    ax.scatter(*p_goal, marker="*", s=180, color="red",
               label="TCP goal", zorder=6)

    # ── 4. 접근 방향 화살표 (마젠타) ───────────
    v_approach = approach_dir * approach_len
    ax.quiver(
        p_goal[0], p_goal[1], p_goal[2],
        v_approach[0], v_approach[1], v_approach[2],
        color="m", linewidth=3.5, arrow_length_ratio=0.25,
    )
    tip = p_goal + v_approach
    ax.text(tip[0], tip[1], tip[2], "  approach\n  (TCP -Y)",
            color="m", fontsize=8)

    # ── 5. pregrasp 위치 (하늘색) ──────────────
    ax.scatter(*p_pre, marker="x", s=100, color="deepskyblue",
               linewidths=2, label="pregrasp", zorder=5)
    ax.plot(
        [p_pre[0], p_goal[0]],
        [p_pre[1], p_goal[1]],
        [p_pre[2], p_goal[2]],
        linestyle="-", linewidth=1.8, color="deepskyblue",
    )

    # ── 6. 현재 TCP → goal 이동 경로 (초록 점선)
    ax.plot(
        [p_cur[0], p_goal[0]],
        [p_cur[1], p_goal[1]],
        [p_cur[2], p_goal[2]],
        linestyle="--", linewidth=1.8, color="limegreen",
        label="cur → goal",
    )

    # ── 축 범위 자동 조정 ──────────────────────
    pts = np.vstack([p_cur, p_obj, p_goal, p_pre, tip])
    margin = max(axis_len, approach_len) * 1.5
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    set_axes_equal_3d(ax)

    ax.legend(loc="upper left", fontsize=8)
    fig_store["fig"].tight_layout()

    if save_dir:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"grasp_{cls}.png"
        fig_store["fig"].savefig(str(fname), dpi=150)
        print(f"[VIZ] saved: {fname}")

    plt.show(block=False)
    plt.pause(0.001)


# ============================================================
# ROS2 Test Node
# ============================================================

class SixDPoseTransformTestNode(Node):
    """
    /object_poses 수신 → 로봇 현재 TCP 쿼리 → 좌표 변환 → 시각화

    트리거 토픽 없음. object_poses 메시지가 올 때마다 동작.
    실제 publish 없음 (viz only).
    """

    def __init__(self):
        super().__init__("sixd_pose_transform_test_node")

        # ── params ──────────────────────────────────
        self.declare_parameter("robot_ip",                  "192.168.1.10")
        self.declare_parameter("use_robot",                 True)
        self.declare_parameter("object_topic",              "/object_poses")
        self.declare_parameter("handeye_result_path",       "")
        self.declare_parameter("object_grasp_yaml_path",    "")
        self.declare_parameter("min_confidence",            0.3)
        self.declare_parameter("canonicalize_object_axes",  True)
        self.declare_parameter("canonicalize_z_flip_margin", 0.05)
        # X/Y canonicalization: X/Y 중 더 수평인 축을 object +X로 선택
        self.declare_parameter("canonicalize_xy_flatter_as_x", True)
        self.declare_parameter("canonicalize_xy_swap_margin", 0.03)
        self.declare_parameter("canonicalize_xy_max_flatness", 0.85)
        self.declare_parameter("visualize_axes_length_mm",  50.0)
        self.declare_parameter("visualize_approach_length_mm", 80.0)
        self.declare_parameter("visualize_save_dir",        "")

        # RB5 마지막 joint 후보 필터 기준
        # peg_camera_joint의 마지막 joint J5 = 34.16 deg 기준, 허용 범위 = 기준 ±90 deg
        # 전체 joint 리스트는 쓰지 않고, 마지막 joint 기준값만 사용한다.
        self.declare_parameter("reference_last_joint_deg", 34.16)
        self.declare_parameter("last_joint_limit_delta_deg", 95.0)

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.use_robot      = bool(self.get_parameter("use_robot").value)
        self.robot_ip       = str(self.get_parameter("robot_ip").value)

        self.reference_last_joint_deg = float(self.get_parameter("reference_last_joint_deg").value)
        self.last_joint_limit_delta_deg = float(self.get_parameter("last_joint_limit_delta_deg").value)

        # ── class maps ──────────────────────────────
        self.class_to_id = {
            "cylinder": 0, "cylinder_insert": 0,
            "hole":     1, "hole_insert":     1,
            "cross":    2, "cross_insert":    2,
        }

        # ── load config ─────────────────────────────
        self.ee_T_cam   = self._load_handeye()
        self.grasp_cfg  = self._load_grasp_yaml()

        # ── rbpodo robot data ────────────────────────
        self._robot_data = None
        if self.use_robot:
            self._init_robot_data()

        # ── viz state ───────────────────────────────
        self._fig_store: dict = {}

        # ── ROS sub ─────────────────────────────────
        obj_topic = str(self.get_parameter("object_topic").value)
        self.create_subscription(String, obj_topic, self._object_callback, 10)

        self.get_logger().info(
            f"SixDPoseTransformTestNode ready | "
            f"object_topic={obj_topic} | "
            f"use_robot={self.use_robot} | "
            f"robot_ip={self.robot_ip}"
        )

    # ── robot data init ──────────────────────────────

    def _init_robot_data(self):
        try:
            import rbpodo as rb
            self._robot_data = rb.CobotData(self.robot_ip)
            self.get_logger().info(f"[ROBOT] CobotData connected: {self.robot_ip}")
        except Exception as e:
            self.get_logger().warn(f"[ROBOT] CobotData init failed: {e}  → fallback to dummy TCP")
            self._robot_data = None

    # ── TCP 쿼리 ────────────────────────────────────

    def _query_current_tcp_mm(self) -> Optional[np.ndarray]:
        """
        현재 로봇 TCP pose6 (mm, deg) 반환.
        use_robot=False 또는 연결 실패 시 더미값 반환.
        """
        if self.use_robot and self._robot_data is not None:
            pose = get_current_tcp_pose_vec_mm(self._robot_data)
            if pose is not None:
                self.get_logger().info(
                    f"[TCP] current={np.round(pose, 2).tolist()}"
                )
                return pose
            self.get_logger().warn("[TCP] request_data() returned None, using dummy")

        # 더미: 로봇 정면 위쪽에 있다고 가정
        dummy = np.array([0.0, -400.0, 350.0, 90.0, 0.0, 45.0], dtype=np.float64)
        self.get_logger().info(f"[TCP] dummy={dummy.tolist()}")
        return dummy

    # ── config loading ───────────────────────────────

    def _load_handeye(self) -> np.ndarray:
        param = str(self.get_parameter("handeye_result_path").value).strip()
        if param:
            path = Path(param)
        else:
            path = (
                Path(get_package_share_directory("calib"))
                / "config" / "handeye_capture_rs" / "handeye_result.json"
            )
        if not path.exists():
            raise FileNotFoundError(f"handeye_result.json not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "ee_T_cam" not in data:
            raise KeyError(f"'ee_T_cam' not found in {path}")
        T_m = np.asarray(data["ee_T_cam"], dtype=np.float64).reshape(4, 4)
        T_m[:3, :3] = orthonormalize_R(T_m[:3, :3])
        validate_T(T_m, name="ee_T_cam_m")
        T_mm = T_m_to_T_mm(T_m)
        validate_T(T_mm, name="ee_T_cam_mm")
        self.get_logger().info(f"[HANDEYE] loaded: {path}")
        return T_mm

    def _load_grasp_yaml(self) -> Dict[str, Dict[str, Any]]:
        yaml_path = str(self.get_parameter("object_grasp_yaml_path").value).strip()
        if not yaml_path:
            self.get_logger().warn("[YAML] object_grasp_yaml_path empty → identity transforms")
            return {}
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"object_grasp_yaml_path not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        global_unit = data.get("unit", "mm")
        objects = data.get("objects", {}) or {}
        if not isinstance(objects, dict):
            raise ValueError("object_grasp_yaml: 'objects' must be a dict.")

        cfg: Dict[str, Dict[str, Any]] = {}

        def _pick(obj_data, keys, default):
            for key in keys:
                blk = obj_data.get(key)
                if isinstance(blk, dict) and "matrix" in blk:
                    unit = blk.get("unit", obj_data.get("unit", global_unit))
                    return matrix_from_data(blk["matrix"], unit=unit)
                elif blk is not None:
                    return matrix_from_data(blk, unit=obj_data.get("unit", global_unit))
            return default.copy()

        for name, obj_data in objects.items():
            obj_data = obj_data or {}
            if not isinstance(obj_data, dict):
                raise ValueError(f"object_grasp_yaml: objects.{name} must be a dict.")

            # 1번 코드와 동일: raw/CAD object frame -> centered object frame
            raw_T_center = _pick(
                obj_data,
                ["object_to_center", "object_to_canonical", "object_to_grasp"],
                np.eye(4, dtype=np.float64),
            )

            # centered object frame -> final TCP goal frame
            center_T_tcp = _pick(
                obj_data,
                ["centered_object_to_tcp", "center_to_tcp", "canonical_to_tcp", "object_center_to_tcp"],
                np.eye(4, dtype=np.float64),
            )

            cfg[str(name)] = {
                "raw_object_T_centered_object": raw_T_center,
                "centered_object_T_tcp_goal": center_T_tcp,
                "symmetry": obj_data.get("symmetry", {}) or {},
                "tilted_grasp": obj_data.get("tilted_grasp", {}) or {},
                "cylinder_yaw_search": obj_data.get("cylinder_yaw_search", {}) or {},
            }
        self.get_logger().info(f"[YAML] loaded: {path}")
        return cfg

    def _get_transforms(
        self,
        cls: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        base = canonical_object_name(cls)
        item = self.grasp_cfg.get(cls) or self.grasp_cfg.get(base) or {
            "raw_object_T_centered_object": np.eye(4, dtype=np.float64),
            "centered_object_T_tcp_goal": np.eye(4, dtype=np.float64),
            "symmetry": {},
            "tilted_grasp": {},
            "cylinder_yaw_search": {},
        }
        raw_T_c = np.asarray(item["raw_object_T_centered_object"], dtype=np.float64).reshape(4, 4)
        center_T_tcp = np.asarray(item["centered_object_T_tcp_goal"], dtype=np.float64).reshape(4, 4)
        symmetry = item.get("symmetry", {}) or {}
        tilted_grasp = item.get("tilted_grasp", {}) or {}
        cylinder_yaw_search = item.get("cylinder_yaw_search", {}) or {}
        validate_T(raw_T_c, name=f"raw_T_center[{cls}]")
        validate_T(center_T_tcp, name=f"center_T_tcp[{cls}]")
        return (
            raw_T_c.copy(),
            center_T_tcp.copy(),
            dict(symmetry),
            dict(tilted_grasp),
            dict(cylinder_yaw_search),
        )

    def _make_symmetric_grasp_candidates(
        self,
        center_T_tcp_nominal: np.ndarray,
        symmetry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        centered_object_to_tcp 기본 grasp를 object +Z 기준으로 0/90/180/270... 회전시켜 후보 생성.
        """
        yaw_candidates = symmetry.get("yaw_candidates_deg", [0.0])
        candidates: List[Dict[str, Any]] = []

        for yaw in yaw_candidates:
            yaw = float(yaw)
            center_T_tcp = T_rot_z_deg(yaw) @ center_T_tcp_nominal
            validate_T(center_T_tcp, name=f"center_T_tcp_candidate[yaw={yaw}]")
            candidates.append({
                "name": f"yaw_{yaw:.0f}",
                "yaw_deg": yaw,
                "center_T_tcp": center_T_tcp,
            })

        if not candidates:
            candidates.append({
                "name": "yaw_0",
                "yaw_deg": 0.0,
                "center_T_tcp": center_T_tcp_nominal.copy(),
            })

        return candidates

    def _estimate_last_joint_for_goal(
        self,
        base_T_goal: np.ndarray,
        reference_T_tcp: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        peg_camera_joint의 마지막 joint J5=reference_last_joint_deg 기준으로,
        목표 TCP 자세가 J5를 얼마나 더 돌려야 하는지 근사 추정한다.

        핵심 단순화:
          - 전체 peg_camera_joint 리스트는 쓰지 않는다.
          - 현재 카메라 자세의 TCP 방향(reference_T_tcp)을 기준 자세로 사용한다.
          - 목표 TCP +X가 기준 TCP +X에서 얼마나 회전했는지를 TCP +Y축 기준 signed angle로 본다.
          - 그 회전량을 J5 변화량으로 근사한다.

        return:
          estimated_j5_deg, delta_from_reference_abs_deg, signed_delta_deg
        """
        R_ref = orthonormalize_R(reference_T_tcp[:3, :3])
        R_goal = orthonormalize_R(base_T_goal[:3, :3])

        ref_x = R_ref[:, 0]
        goal_x = R_goal[:, 0]
        goal_y = R_goal[:, 1]  # TCP +Y, 즉 pregrasp/retreat 축

        signed_delta = signed_angle_about_axis_deg(ref_x, goal_x, goal_y)
        estimated_j5 = wrap_deg(self.reference_last_joint_deg + signed_delta)
        delta_abs = abs(wrap_deg(estimated_j5 - self.reference_last_joint_deg))
        return estimated_j5, delta_abs, signed_delta

    # ── main callback ────────────────────────────────

    def _object_callback(self, msg: String):
        # 1. JSON 파싱
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[CB] JSON parse failed: {e}")
            return

        # triggered 메시지만 처리 (preview는 objects 비어있음)
        objects = data.get("objects") or []
        if not objects and isinstance(data.get("target"), dict):
            objects = [data["target"]]
        if not objects:
            return

        # 2. 현재 TCP 쿼리 (object_poses 수신 시점)
        current_tcp_pose6 = self._query_current_tcp_mm()
        if current_tcp_pose6 is None:
            self.get_logger().warn("[CB] TCP query failed, skip")
            return
        base_T_ee = pose6_mm_deg_to_T_mm(current_tcp_pose6)

        # 3. 각 감지 물체에 대해 변환 + 시각화
        for obj in objects:
            target = self._compute_target(obj, base_T_ee)
            if target is None:
                continue

            self.get_logger().info(
                f"[RESULT] class={target['class']} "
                f"conf={target['confidence']:.3f} "
                f"z_flipped={target.get('axis_info', {}).get('z_flipped', '?')} "
                f"grasp={target.get('selected_grasp_name', '?')} "
                f"yaw={target.get('selected_grasp_yaw_deg', '?')} "
                f"est_J5={target.get('estimated_last_joint_deg', 0.0):.2f} "
                f"dJ5={target.get('last_joint_delta_deg', 0.0):.2f} "
                f"pose6={np.round(target['target_pose6'], 2).tolist()}"
            )

            # 메인 스레드 큐에 전달 (matplotlib은 메인 스레드 전용)
            params = {
                "axis_len":   float(self.get_parameter("visualize_axes_length_mm").value),
                "approach_len": float(self.get_parameter("visualize_approach_length_mm").value),
                "save_dir":   str(self.get_parameter("visualize_save_dir").value).strip(),
            }
            if hasattr(self, "_viz_queue"):
                try:
                    self._viz_queue.put_nowait((target, current_tcp_pose6, params))
                except Exception:
                    self.get_logger().warn("[VIZ] queue full, drop frame")

    def _compute_target(
        self,
        obj: Dict[str, Any],
        base_T_ee: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        1번 코드의 object pose -> centered object -> safe centered object -> TCP goal 로직을
        테스트 노드에 그대로 이식한 버전.

        테스트 노드 차이점:
          - base_T_ee는 trigger TCP가 아니라, /object_poses 수신 시점에 로봇에서 직접 쿼리한 현재 TCP이다.
          - publish는 하지 않고 target dict를 반환해서 matplotlib 시각화에 사용한다.
        """
        cls = str(obj.get("class", ""))
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            self.get_logger().info(f"[SKIP] {cls} conf={conf:.3f} < {self.min_confidence}")
            return None
        if cls not in self.class_to_id:
            self.get_logger().warn(f"[SKIP] unknown class: {cls}")
            return None

        obj_id = int(self.class_to_id[cls])

        try:
            # 1. FoundationPose/CAD raw object pose in camera frame.
            cam_T_obj = object_json_to_cam_T_obj_mm(obj)

            # 2. 현재 TCP/EE pose + hand-eye + camera object pose.
            #    1번 코드의 base_T_ee @ ee_T_cam @ cam_T_obj와 동일한 변환식이다.
            base_T_obj_raw = base_T_ee @ self.ee_T_cam @ cam_T_obj
            validate_T(base_T_obj_raw, name=f"base_T_obj_raw[{cls}]")

            # 3. YAML transform 로드:
            #    raw object/CAD frame -> centered object frame
            #    centered object frame -> nominal TCP goal frame
            raw_T_center, center_T_tcp_nominal, symmetry, tilted_grasp_cfg, cylinder_yaw_cfg = (
                self._get_transforms(cls)
            )

            base_T_center = base_T_obj_raw @ raw_T_center
            validate_T(base_T_center, name=f"base_T_centered_object[{cls}]")

            # 4. Z-up defense: object +Z가 world +Z에 더 가깝도록 보정.
            axis_info = None
            if bool(self.get_parameter("canonicalize_object_axes").value):
                base_T_center, axis_info = defend_centered_object_z_up(
                    base_T_center,
                    z_flip_margin=float(self.get_parameter("canonicalize_z_flip_margin").value),
                )
                if axis_info["z_flipped"]:
                    self.get_logger().info(
                        f"[Z_DEFENSE] cls={cls} z_flipped={axis_info['z_flipped']} "
                        f"keep_dot_up={axis_info.get('keep_dot_up')} "
                        f"flip_dot_up={axis_info.get('flip_dot_up')} "
                        f"after_dot_up={axis_info['after_dot_up']}"
                    )

            # 5. XY_FLAT + hole/cross tilted branch.
            xy_info = None
            tilted_grasp_used = False
            tilted_grasp_reason = ""
            tilted_pre_yaw_deg = 0.0
            tilt_info = compute_centered_axis_tilt_info(base_T_center)

            if bool(self.get_parameter("canonicalize_xy_flatter_as_x").value):
                base_T_center, xy_info = canonicalize_xy_flatter_as_x(
                    base_T_center,
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

                tilt_info = compute_centered_axis_tilt_info(base_T_center)
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

                    # centered object local +Z 기준 pre_yaw_deg 회전.
                    base_T_center = base_T_center @ T_rot_z_deg(tilted_pre_yaw_deg)
                    validate_T(base_T_center, name=f"base_T_centered_object_tilted_yaw[{cls}]")

                    # 회전 후 다시 XY_FLAT 수행.
                    base_T_center, xy_info_2 = canonicalize_xy_flatter_as_x(
                        base_T_center,
                        swap_margin=float(self.get_parameter("canonicalize_xy_swap_margin").value),
                    )
                    tilt_info_2 = compute_centered_axis_tilt_info(base_T_center)

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

            # 6. symmetry.yaw_candidates_deg 후보 생성 후 J5 limit 필터링/선택.
            grasp_candidates = self._make_symmetric_grasp_candidates(
                center_T_tcp_nominal,
                symmetry,
            )

            best = None
            is_cylinder = canonical_object_name(cls) == "cylinder"
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # Cylinder-specific policy:
            #   enable=true  → J5 통과 후보 중 object_x 평평함 우선, 비슷하면 J5 delta 최소
            #   enable=false → J5 통과 후보 중 J5 delta 최소
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
                    # TCP 축이 아니라 물체/centered-object 축 기준이다.
                    base_T_centered_yaw = base_T_centered @ T_rot_z_deg(float(yaw_deg))
                    R_eval = orthonormalize_R(base_T_centered_yaw[:3, :3])
                    return R_eval[:, object_axis_map[axis_name]]

                if axis_name in tcp_axis_map:
                    # 실험용 fallback: 최종 TCP frame의 축 평가.
                    R_eval = orthonormalize_R(base_T_tcp[:3, :3])
                    return R_eval[:, tcp_axis_map[axis_name]]

                self.get_logger().warn(
                    f"[CYL_SCORE_AXIS] unknown score_axis='{axis_name}', fallback to object_x"
                )
                base_T_centered_yaw = base_T_centered @ T_rot_z_deg(float(yaw_deg))
                R_eval = orthonormalize_R(base_T_centered_yaw[:3, :3])
                return R_eval[:, 0]

            for cand in grasp_candidates:
                base_T_goal_cand = base_T_center @ cand["center_T_tcp"]
                validate_T(base_T_goal_cand, name=f"base_T_goal[{cls}:{cand['name']}]")

                estimated_j5, delta_j5, signed_delta_j5 = self._estimate_last_joint_for_goal(
                    base_T_goal_cand,
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
                        base_T_center,
                        base_T_goal_cand,
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
                        "base_T_goal": base_T_goal_cand,
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
                    f"[NO_VALID_GRASP] cls={cls}: all candidates exceed "
                    f"J5 reference {self.reference_last_joint_deg:.2f} "
                    f"±{self.last_joint_limit_delta_deg:.1f} deg"
                )
                return None

            selected = best["candidate"]
            base_T_goal = best["base_T_goal"]
            pose6 = T_mm_to_pose6_mm_deg(base_T_goal)
            p = base_T_goal[:3, 3]

            output_obj_id = obj_id
            if tilted_grasp_used and bool(tilted_grasp_cfg.get("mark_output_id_negative", True)):
                output_obj_id = -abs(obj_id)

            return {
                "class": cls,
                "id": output_obj_id,
                "raw_id": obj_id,
                "tilted_grasp_used": bool(tilted_grasp_used),
                "tilted_grasp_reason": tilted_grasp_reason,
                "tilted_pre_yaw_deg": float(tilted_pre_yaw_deg),
                "confidence": conf,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "target_pose6": pose6,
                "target_T": base_T_goal,
                "base_T_obj_raw": base_T_obj_raw,
                "base_T_centered_object_safe": base_T_center,
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
                "output_format": "viz_only_id_plus_moveL_pose6_7",
            }

        except Exception as e:
            self.get_logger().warn(f"[ERROR] compute_target cls={cls}: {e}")
            traceback.print_exc()
            return None


# ============================================================
# main
# ============================================================

def main(args=None):
    """
    matplotlib GUI 는 메인 스레드에서만 동작해야 한다.
    ROS spin 은 별도 스레드에서 실행하고,
    콜백이 큐에 넣은 데이터를 메인 스레드 루프에서 꺼내 그린다.
    """
    import threading
    import queue as _queue
    import matplotlib
    matplotlib.use("TkAgg")   # 헤드리스 환경이면 "Agg" 로 변경
    import matplotlib.pyplot as plt

    rclpy.init(args=args)
    node = SixDPoseTransformTestNode()

    # 콜백 → 메인 스레드 데이터 전달 큐
    viz_q: _queue.Queue = _queue.Queue(maxsize=4)
    node._viz_queue = viz_q

    # ROS spin 스레드 (daemon=True → 메인 종료 시 자동 종료)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig_store: dict = {}
    print("[MAIN] matplotlib 루프 시작 — Ctrl+C 로 종료")

    try:
        while rclpy.ok():
            try:
                item = viz_q.get(timeout=0.05)
            except _queue.Empty:
                plt.pause(0.02)   # GUI 이벤트 처리 (창 유지 핵심)
                continue

            target, tcp_pose6, params = item
            visualize_grasp_target(
                target=target,
                current_tcp_pose6=tcp_pose6,
                axis_len=params["axis_len"],
                approach_len=params["approach_len"],
                save_dir=params["save_dir"],
                fig_store=fig_store,
            )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()