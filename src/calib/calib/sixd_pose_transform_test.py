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
    T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    validate_T(T, name="z_defense_in")
    up = normalize_vec(up)
    R = orthonormalize_R(T[:3, :3])
    x, y, z = R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()
    before = {"x": float(np.dot(x, up)), "y": float(np.dot(y, up)), "z": float(np.dot(z, up))}
    z_flipped = False
    if before["z"] < -float(z_flip_margin):
        x, z = -x, -z
        z_flipped = True
    z = normalize_vec(z)
    x = normalize_vec(x - z * float(np.dot(x, z)))
    y = normalize_vec(np.cross(z, x))
    T[:3, :3] = orthonormalize_R(np.column_stack([x, y, z]))
    validate_T(T, name="z_defense_out")
    info = {
        "z_flipped": z_flipped,
        "before_dot_up": before,
        "after_dot_up": {
            "x": float(np.dot(T[:3, 0], up)),
            "y": float(np.dot(T[:3, 1], up)),
            "z": float(np.dot(T[:3, 2], up)),
        },
    }
    return T, info


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
        self.declare_parameter("visualize_axes_length_mm",  50.0)
        self.declare_parameter("visualize_approach_length_mm", 80.0)
        self.declare_parameter("visualize_save_dir",        "")

        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.use_robot      = bool(self.get_parameter("use_robot").value)
        self.robot_ip       = str(self.get_parameter("robot_ip").value)

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

        for name, obj_data in (data.get("objects", {}) or {}).items():
            obj_data = obj_data or {}
            raw_T_center = _pick(
                obj_data,
                ["object_to_center", "object_to_canonical", "object_to_grasp"],
                np.eye(4, dtype=np.float64),
            )
            center_T_tcp = _pick(
                obj_data,
                ["centered_object_to_tcp", "center_to_tcp", "canonical_to_tcp"],
                np.eye(4, dtype=np.float64),
            )
            cfg[str(name)] = {
                "raw_object_T_centered_object": raw_T_center,
                "centered_object_T_tcp_goal":   center_T_tcp,
            }
        self.get_logger().info(f"[YAML] loaded: {path}")
        return cfg

    def _get_transforms(self, cls: str) -> Tuple[np.ndarray, np.ndarray]:
        base = canonical_object_name(cls)
        item = self.grasp_cfg.get(cls) or self.grasp_cfg.get(base) or {
            "raw_object_T_centered_object": np.eye(4, dtype=np.float64),
            "centered_object_T_tcp_goal":   np.eye(4, dtype=np.float64),
        }
        raw_T_c   = np.asarray(item["raw_object_T_centered_object"], dtype=np.float64).reshape(4, 4)
        center_T_tcp = np.asarray(item["centered_object_T_tcp_goal"],    dtype=np.float64).reshape(4, 4)
        validate_T(raw_T_c,      name=f"raw_T_center[{cls}]")
        validate_T(center_T_tcp, name=f"center_T_tcp[{cls}]")
        return raw_T_c.copy(), center_T_tcp.copy()

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
        cls  = str(obj.get("class", ""))
        conf = float(obj.get("confidence", 0.0))

        if conf < self.min_confidence:
            self.get_logger().info(f"[SKIP] {cls} conf={conf:.3f} < {self.min_confidence}")
            return None
        if cls not in self.class_to_id:
            self.get_logger().warn(f"[SKIP] unknown class: {cls}")
            return None

        try:
            # cam → base 변환
            cam_T_obj   = object_json_to_cam_T_obj_mm(obj)
            base_T_raw  = base_T_ee @ self.ee_T_cam @ cam_T_obj
            validate_T(base_T_raw, name=f"base_T_raw[{cls}]")

            # YAML 오프셋
            raw_T_center, center_T_tcp = self._get_transforms(cls)
            base_T_center = base_T_raw @ raw_T_center
            validate_T(base_T_center, name=f"base_T_center[{cls}]")

            # Z-up defense
            axis_info = None
            if bool(self.get_parameter("canonicalize_object_axes").value):
                base_T_center, axis_info = defend_centered_object_z_up(
                    base_T_center,
                    z_flip_margin=float(self.get_parameter("canonicalize_z_flip_margin").value),
                )
                if axis_info["z_flipped"]:
                    self.get_logger().info(
                        f"[DEFENSE] z_flipped cls={cls} "
                        f"before={axis_info['before_dot_up']} "
                        f"after={axis_info['after_dot_up']}"
                    )

            # 최종 TCP goal
            base_T_goal = base_T_center @ center_T_tcp
            validate_T(base_T_goal, name=f"base_T_goal[{cls}]")

            pose6 = T_mm_to_pose6_mm_deg(base_T_goal)

            return {
                "class":                      cls,
                "id":                         int(self.class_to_id[cls]),
                "confidence":                 conf,
                "target_pose6":               pose6,
                "target_T":                   base_T_goal,
                "base_T_obj_raw":             base_T_raw,
                "base_T_centered_object_safe": base_T_center,
                "axis_info":                  axis_info,
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