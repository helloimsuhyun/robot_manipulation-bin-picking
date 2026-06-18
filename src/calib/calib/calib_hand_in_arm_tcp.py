import time
import traceback
import cv2
#cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import numpy as np
import rbpodo as rb
from pathlib import Path

from calibration import RealSenseCharucoHandEye

SAVE_DIR = Path(__file__).resolve().parents[1] / "config" / "handeye_capture_rs"


# 예시:
# -117.99 -461.92 380.00 90.04 -0.06 44.45
# -> x(mm) y(mm) z(mm) rx(deg) ry(deg) rz(deg)

ROBOT_IP = "192.168.1.10"

USE_ROBOT = True
RUN_MODE = "real"   # "sim" or "real"
USE_CAMERA = True
SIM_MODE = (RUN_MODE == "sim")

AUTO_MOVE_CAPTURE = True
AUTO_RUN_CALIBRATION_AFTER_CAPTURE = False

TCP_POS_TOL_MM = 2.0
TCP_RPY_TOL_DEG = 2.0
TCP_REACH_TIMEOUT_S = 30.0
TCP_STABLE_COUNT = 5
TCP_POLL_DT = 0.05

MOVE_SPEED = 150.0
MOVE_ACCEL = 150.0
CAPTURE_DELAY_S = 0.7

# camera fwd = local -y ...
CAMERA_FWD_AXIS = 1    # local_y = R[:, 1]
CAMERA_FWD_SIGN = -1   # 음의 방향

RADIUS_MM = 200.0
GRID_DIST_OFFSETS_MM = [0.0, 80.0, 160.0]
GRID_U_OFFSETS_MM = [-100.0, -50.0, 0.0, 50.0, 100.0]
GRID_V_OFFSETS_MM = [-60.0, 0.0, 60.0]
MIN_SAFE_Z_MM = 150.0




# ──────────────────────────────────────────────
# util
# ──────────────────────────────────────────────

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_trajectory_3d(pose_list, initial_pose_vec, radius_mm):
    center_mm = compute_center_from_initial_pose(initial_pose_vec, radius_mm)

    pts = np.array([p[:3] for p in pose_list], dtype=np.float64)
    init = np.array(initial_pose_vec[:3], dtype=np.float64)

    fig = plt.figure("Trajectory 3D Preview")
    ax = fig.add_subplot(111, projection='3d')

    # -----------------------------
    # 점 위치
    # -----------------------------
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        s=40,
        label="poses"
    )

    # 이동 순서 라인
    ax.plot(
        pts[:, 0], pts[:, 1], pts[:, 2],
        linewidth=1
    )

    # -----------------------------
    # 각 점에서 EE / Camera Forward 방향 화살표
    # -----------------------------
    for i, p in enumerate(pose_list):
        T = pose_vec_to_T(p)
        pos = T[:3, 3]
        R = T[:3, :3]

        fwd = get_camera_fwd(R)

        ax.quiver(
            pos[0], pos[1], pos[2],     # 시작점
            fwd[0], fwd[1], fwd[2],     # 방향벡터
            length=40,                  # 화살표 길이(mm 느낌)
            normalize=True,
            arrow_length_ratio=0.25
        )

    # -----------------------------
    # board center
    # -----------------------------
    ax.scatter(
        [center_mm[0]],
        [center_mm[1]],
        [center_mm[2]],
        s=140,
        marker='x',
        label="board center"
    )

    # -----------------------------
    # initial pose
    # -----------------------------
    ax.scatter(
        [init[0]],
        [init[1]],
        [init[2]],
        s=90,
        marker='^',
        label="initial"
    )

    # 번호 표시
    for i, p in enumerate(pts):
        ax.text(
            p[0], p[1], p[2],
            str(i),
            size=8
        )

    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.set_title("World Trajectory Preview")
    ax.legend()

    plt.show()

def debug(msg):
    print(f"[DEBUG] {msg}", flush=True)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n >= eps else v.copy()


def euler_zyx_deg_to_R(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def R_to_euler_zyx_deg(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    if sy >= 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    return np.degrees([rx, ry, rz])


def pose_vec_to_T(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    x, y, z, rx, ry, rz = pose6

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_zyx_deg_to_R(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def T_to_pose_vec(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    rx, ry, rz = R_to_euler_zyx_deg(T[:3, :3])
    return np.array([T[0, 3], T[1, 3], T[2, 3], rx, ry, rz], dtype=np.float64)


def base_T_ee_from_move_l_pose(pose6: np.ndarray) -> np.ndarray:
    T = pose_vec_to_T(pose6).copy()
    T[:3, 3] /= 1000.0   # mm → m
    return T


def get_camera_fwd(R0: np.ndarray) -> np.ndarray:
    axis_vec = R0[:, CAMERA_FWD_AXIS]
    return normalize(CAMERA_FWD_SIGN * axis_vec)


# ──────────────────────────────────────────────
# trajectory generation
# ──────────────────────────────────────────────

def make_fwd_perp_axes(fwd: np.ndarray):
    """fwd에 수직인 평면 축 2개 생성."""
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(fwd, up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    px = normalize(np.cross(up, fwd))
    py = normalize(np.cross(fwd, px))
    return px, py


def make_look_at_R(cam_pos_mm, target_mm, ref_up, ref_up_fallback):
    """
    cam_pos에서 target을 바라보는 EE rotation 행렬.
    카메라 forward = EE -local_y
    """
    view_dir = normalize(target_mm - cam_pos_mm)  # 카메라 광축 방향
    local_y = -view_dir                           # EE -local_y = view_dir

    up = ref_up if abs(np.dot(local_y, ref_up)) < 0.98 else ref_up_fallback

    local_z = normalize(np.cross(up, local_y))
    local_x = normalize(np.cross(local_y, local_z))
    local_z = normalize(np.cross(local_x, local_y))

    # 기본 look-at 자세
    R = np.column_stack([local_x, local_y, local_z])

    # camera forward = -local_y 이므로,
    # 보드 보는 방향은 유지하면서 local_y 축 주변으로 roll만 회전
    ROLL_OFFSET_DEG = 25.0
    a = np.radians(ROLL_OFFSET_DEG)

    Ry_local = np.array([
        [ np.cos(a), 0, np.sin(a)],
        [0,          1, 0         ],
        [-np.sin(a), 0, np.cos(a)]
    ], dtype=np.float64)

    return R @ Ry_local


def compute_center_from_initial_pose(initial_pose_vec, radius_mm):
    T0 = pose_vec_to_T(initial_pose_vec)
    p0 = T0[:3, 3]
    fwd = get_camera_fwd(T0[:3, :3])
    return p0 + radius_mm * fwd


def build_multi_plane_grid_pose_list(initial_pose_vec, base_dist_mm,
                                     dist_offsets_mm,
                                     x_offsets_mm, z_offsets_mm):
    """
    보드 중심(center_mm)을 기준으로
    near / mid / far 여러 평면에서 3x3 grid 포즈 생성.
    각 포즈의 카메라 forward(-local_y)는 항상 보드 중심을 바라본다.
    """
    T0 = pose_vec_to_T(initial_pose_vec)
    p0 = T0[:3, 3]
    fwd = get_camera_fwd(T0[:3, :3])

    center_mm = p0 + base_dist_mm * fwd

    perp_x, perp_y = make_fwd_perp_axes(fwd)

    ref_up = perp_y
    ref_up_fallback = perp_x

    pose_list = []

    for d_off in dist_offsets_mm:
        dist_mm = base_dist_mm + d_off
        if dist_mm <= 0:
            continue

        for dz in z_offsets_mm:
            for dx in x_offsets_mm:
                cam_pos_mm = center_mm - dist_mm * fwd + dx * perp_x + dz * perp_y

                R = make_look_at_R(
                    cam_pos_mm=cam_pos_mm,
                    target_mm=center_mm,
                    ref_up=ref_up,
                    ref_up_fallback=ref_up_fallback,
                )

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = cam_pos_mm
                pose_list.append(T_to_pose_vec(T))

    return pose_list


# ──────────────────────────────────────────────
# 입력 헬퍼
# ──────────────────────────────────────────────

def input_initial_pose_vec():
    print("\n[INPUT INITIAL POSE]")
    print("format: x(mm) y(mm) z(mm) rx(deg) ry(deg) rz(deg)")
    print("example: -84.45 -470.73 194.46 90.00 0.00 43.83")

    while True:
        vals = input("initial pose >> ").strip().split()
        if len(vals) != 6:
            print("❌ 6개 입력해야 함")
            continue

        try:
            return np.array(list(map(float, vals)), dtype=np.float64)
        except Exception:
            print("❌ 숫자로 다시 입력해라")


def input_radius_mm(measured_mm):
    print(f"\n현재 보드까지 측정 거리: {measured_mm:.1f} mm")
    print("원하는 radius를 입력하거나 엔터 치면 측정값 사용")

    while True:
        s = input(f"radius mm [{measured_mm:.1f}] >> ").strip()
        if s == "":
            return float(measured_mm)

        try:
            return float(s)
        except ValueError:
            print("❌ 숫자로 입력하거나 그냥 엔터 쳐라")


# ──────────────────────────────────────────────
# safety / robot move
# ──────────────────────────────────────────────

def is_pose_safe_basic(pose_vec):
    x, y, z, rx, ry, rz = pose_vec

    if z < MIN_SAFE_Z_MM:
        return False, f"z too low: {z:.1f} < {MIN_SAFE_Z_MM:.1f}"

    return True, "ok"


def filter_unsafe_poses(pose_list):
    safe_list = []
    removed = []

    for i, pose in enumerate(pose_list):
        ok, reason = is_pose_safe_basic(pose)
        if ok:
            safe_list.append(pose)
        else:
            removed.append((i, pose, reason))

    print(f"[SAFE FILTER] kept={len(safe_list)} removed={len(removed)}")
    for i, pose, reason in removed:
        print(f"  - removed idx={i}: {reason} / pose={pose}")

    return safe_list, removed


def angle_diff_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a - b + 180.0) % 360.0 - 180.0


def get_current_tcp_pose_vec_mm(robot_data):
    state = robot_data.request_data()
    if state is None:
        return None

    if hasattr(state.sdata, "tcp"):
        tcp_info = np.array(state.sdata.tcp, dtype=np.float64)
    elif hasattr(state.sdata, "tcp_pos"):
        tcp_info = np.array(state.sdata.tcp_pos, dtype=np.float64)
    elif hasattr(state.sdata, "cur_pos"):
        tcp_info = np.array(state.sdata.cur_pos, dtype=np.float64)
    else:
        raise RuntimeError("TCP pose field를 찾지 못했습니다.")

    return tcp_info[:6].copy()


def wait_until_tcp_reached(
    robot_data,
    target_pose_vec,
    pos_tol_mm=TCP_POS_TOL_MM,
    rpy_tol_deg=TCP_RPY_TOL_DEG,
    timeout_s=TCP_REACH_TIMEOUT_S,
    stable_count_required=TCP_STABLE_COUNT,
    poll_dt=TCP_POLL_DT,
):
    target_pose_vec = np.asarray(target_pose_vec, dtype=np.float64).reshape(6)

    start = time.time()
    stable_count = 0

    last_cur = None
    last_pos_err = float("inf")
    last_rpy_err = float("inf")

    while time.time() - start < timeout_s:
        cur = get_current_tcp_pose_vec_mm(robot_data)
        if cur is None:
            time.sleep(poll_dt)
            continue

        last_cur = cur

        pos_err = float(np.linalg.norm(cur[:3] - target_pose_vec[:3]))
        rpy_err_vec = angle_diff_deg(cur[3:6], target_pose_vec[3:6])
        rpy_err = float(np.linalg.norm(rpy_err_vec))

        last_pos_err = pos_err
        last_rpy_err = rpy_err

        if pos_err <= pos_tol_mm and rpy_err <= rpy_tol_deg:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_count_required:
            print(
                f"[TCP ARRIVED] pos_err={pos_err:.3f} mm, "
                f"rpy_err={rpy_err:.3f} deg"
            )
            return True, cur, pos_err, rpy_err

        time.sleep(poll_dt)

    print(
        f"[TCP TIMEOUT] target not reached within {timeout_s:.1f}s / "
        f"pos_err={last_pos_err:.3f} mm, rpy_err={last_rpy_err:.3f} deg"
    )
    print(f"  target = {target_pose_vec}")
    print(f"  current = {last_cur}")

    return False, last_cur, last_pos_err, last_rpy_err

def move_l_blocking(robot, rc, robot_data, pose_vec, speed, accel):
    debug("move_l_blocking: send move_l")

    pose_vec = np.asarray(pose_vec, dtype=np.float64).reshape(6)

    robot.move_l(rc, pose_vec, speed, accel)
    rc.error().throw_if_not_empty()

    ok, cur_pose, pos_err, rpy_err = wait_until_tcp_reached(
        robot_data=robot_data,
        target_pose_vec=pose_vec,
    )

    rc.error().throw_if_not_empty()

    if not ok:
        raise RuntimeError(
            "TCP did not reach target pose. Do not capture. "
            f"pos_err={pos_err:.3f} mm, rpy_err={rpy_err:.3f} deg"
        )

    debug("move_l_blocking: done by TCP pose check")
    return cur_pose


# ──────────────────────────────────────────────
# 뷰어 오버레이
# ──────────────────────────────────────────────

def make_overlay(img, text_lines):
    out = img.copy()
    y = 30
    for line in text_lines:
        cv2.putText(
            out, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        y += 28
    return out


def draw_status_flash(img, text, color, remain_ratio):
    out = img.copy()
    h, w = out.shape[:2]
    alpha = 0.25 + 0.35 * max(0.0, min(1.0, remain_ratio))

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ts, _ = cv2.getTextSize(text, font, 2.0, 4)
    tx = max((w - ts[0]) // 2, 20)
    ty = max(h // 2, 60)

    cv2.putText(out, text, (tx, ty), font, 2.0, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(out, text, (tx, ty), font, 2.0, (255, 255, 255), 4, cv2.LINE_AA)

    return out


def project_points_to_minimap(points_xyz, canvas_w, canvas_h, pad=20):
    pts = np.asarray(points_xyz, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    xs, zs = pts[:, 0], pts[:, 2]
    rx = max(xs.max() - xs.min(), 1e-6)
    rz = max(zs.max() - zs.min(), 1e-6)
    s = min((canvas_w - 2 * pad) / rx, (canvas_h - 2 * pad) / rz)
    xm, zm = 0.5 * (xs.max() + xs.min()), 0.5 * (zs.max() + zs.min())

    out = []
    for x, z in zip(xs, zs):
        out.append([
            int(canvas_w * 0.5 + (x - xm) * s),
            int(canvas_h * 0.5 - (z - zm) * s),
        ])

    return np.asarray(out, dtype=np.int32)


def draw_grid_minimap(img, pose_list, current_index,
                      initial_pose_vec, radius_mm,
                      box_w=320, box_h=260, margin=20):
    out = img.copy()
    H, W = out.shape[:2]
    x1 = max(W - box_w - margin, 0)
    y1 = margin
    x2 = min(x1 + box_w, W)
    y2 = min(y1 + box_h, H)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.putText(out, "Grid Minimap", (x1 + 10, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    mx1, my1, mx2, my2 = x1 + 10, y1 + 35, x2 - 10, y2 - 55
    mw, mh = mx2 - mx1, my2 - my1
    cv2.rectangle(out, (mx1, my1), (mx2, my2), (80, 80, 80), 1)

    center_mm = compute_center_from_initial_pose(initial_pose_vec, radius_mm)
    pose_xyz = np.array([p[:3] for p in pose_list], dtype=np.float64)
    all_xyz = np.vstack([pose_xyz, center_mm.reshape(1, 3)])

    proj = project_points_to_minimap(all_xyz, mw, mh, pad=18)
    puv = proj[:-1].copy()
    cuv = proj[-1].copy()

    puv[:, 0] += mx1
    puv[:, 1] += my1
    cuv[0] += mx1
    cuv[1] += my1

    cv2.circle(out, tuple(cuv), 6, (255, 0, 0), -1)
    cv2.putText(out, "C", (cuv[0] + 8, cuv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    if 0 <= current_index < len(puv):
        cv2.line(out, tuple(cuv), tuple(int(v) for v in puv[current_index]),
                 (100, 100, 255), 1, cv2.LINE_AA)

    for i, uv in enumerate(puv):
        uv = tuple(int(v) for v in uv)
        if i < current_index:
            color, r = (0, 180, 0), 5
        elif i == current_index >= 0:
            color, r = (0, 0, 255), 8
        else:
            color, r = (170, 170, 170), 4

        cv2.circle(out, uv, r, color, -1)

        if i == current_index >= 0:
            cv2.putText(out, f"{i}", (uv[0] + 8, uv[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    total_n = len(pose_list)
    done_n = max(0, current_index)
    remain_n = max(total_n - done_n - (1 if current_index >= 0 else 0), 0)

    cv2.putText(out, f"done:{done_n}  remain:{remain_n}  total:{total_n}",
                (x1 + 10, y2 - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

    return out


def draw_trajectory_preview(img, pose_list, initial_pose_vec, radius_mm):
    out = draw_grid_minimap(img, pose_list, -1, initial_pose_vec, radius_mm)
    h, w = out.shape[:2]

    msg1 = "=== TRAJECTORY PREVIEW ==="
    msg2 = "SPACE: start / i: reinput pose / q: quit"

    cv2.putText(out, msg1, (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(out, msg2, (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    if (not USE_ROBOT) and (not USE_CAMERA):
        cv2.putText(out, "TEST MODE (No Robot / No Camera)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)

    return out


def show_sampler_status(
    viewer_window,
    calib,
    pose_list,
    current_index,
    initial_pose_vec,
    radius_mm,
    status_lines=None,
    flash_text=None,
    flash_color=(0, 180, 0),
):
    """자동 샘플링 중 viewer 갱신."""
    status_lines = status_lines or []

    frame = None if calib is None else calib.get_latest_bgr()
    if frame is not None:
        ok, rvec, tvec, debug_img, msg = calib.collector.detect_board_pose(frame, draw=True)
        if debug_img is None:
            debug_img = frame.copy()

        board_dist_text = "board_dist_mm: N/A"
        if ok and tvec is not None:
            tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(-1)
            board_dist_mm_cur = float(np.linalg.norm(tvec_arr) * 1000.0)
            board_dist_text = f"board_dist_mm: {board_dist_mm_cur:.1f}"

        num_samples = len(calib.collector.samples)
        lines = [
            f"charuco: {'OK' if ok else 'FAIL'} ({msg})",
            board_dist_text,
            f"saved samples: {num_samples}",
            f"current index: {current_index} / total: {len(pose_list)}",
        ] + status_lines + [
            "AUTO MODE: moving + capture / q: stop after current step",
        ]
        show = make_overlay(debug_img, lines)
    else:
        show = np.zeros((720, 1280, 3), np.uint8)
        lines = [
            "AUTO MODE",
            f"current index: {current_index} / total: {len(pose_list)}",
        ] + status_lines + [
            "q: stop after current step",
        ]
        show = make_overlay(show, lines)

    show = draw_grid_minimap(show, pose_list, current_index, initial_pose_vec, radius_mm)

    if flash_text is not None:
        show = draw_status_flash(show, flash_text, flash_color, 1.0)

    cv2.imshow(viewer_window, show)
    key = cv2.waitKey(1) & 0xFF
    return key


def auto_move_and_capture_all(
    robot,
    rc,
    robot_data,
    calib,
    viewer_window,
    pose_list,
    initial_pose_vec,
    radius_mm,
):
    """
    pose_list 전체를 자동으로 순회한다.
    각 pose에서 move_l 완료 판정은 실제 TCP 좌표 기준으로 수행하고,
    calib.capture()에는 명령 target_pose가 아니라 실제 도착 TCP pose를 넣는다.
    """
    print("\n===== AUTO MOVE + CAPTURE START =====")
    print(f"num poses = {len(pose_list)}")
    print("q를 누르면 현재 step 이후 중단합니다.")

    current_index = -1
    success_count = 0
    fail_count = 0
    stop_requested = False

    for next_index, target_pose in enumerate(pose_list):
        current_index = next_index

        key = show_sampler_status(
            viewer_window, calib, pose_list, current_index,
            initial_pose_vec, radius_mm,
            status_lines=[f"AUTO: moving to sample {next_index}"],
            flash_text="MOVING",
            flash_color=(0, 180, 255),
        )
        if key == ord('q') or key == 27:
            stop_requested = True
            break

        print(f"\n[AUTO] move to sample {next_index}/{len(pose_list)-1}")
        print(f"[AUTO TARGET] {target_pose}")

        if not USE_ROBOT:
            arrived_tcp_pose = np.asarray(target_pose, dtype=np.float64).reshape(6)
            res = "virtual move"
            print("[TEST] virtual move")
        else:
            try:
                arrived_tcp_pose = move_l_blocking(
                    robot, rc, robot_data, target_pose,
                    MOVE_SPEED, MOVE_ACCEL
                )
            except Exception as e:
                fail_count += 1
                print(f"[AUTO MOVE FAILED] sample {next_index}: {e}")
                show_sampler_status(
                    viewer_window, calib, pose_list, current_index,
                    initial_pose_vec, radius_mm,
                    status_lines=[f"sample {next_index}: MOVE FAILED"],
                    flash_text="MOVE FAIL",
                    flash_color=(0, 0, 180),
                )
                time.sleep(0.5)
                continue

            print(f"[AUTO ARRIVED TCP] {arrived_tcp_pose}")
            print(f"[AUTO] arrived. wait {CAPTURE_DELAY_S:.1f}s ...")

            t_wait_start = time.time()
            while time.time() - t_wait_start < CAPTURE_DELAY_S:
                key = show_sampler_status(
                    viewer_window, calib, pose_list, current_index,
                    initial_pose_vec, radius_mm,
                    status_lines=[
                        f"sample {next_index}: arrived",
                        f"settling... {time.time() - t_wait_start:.1f}/{CAPTURE_DELAY_S:.1f}s",
                    ],
                    flash_text="ARRIVED",
                    flash_color=(0, 180, 0),
                )
                if key == ord('q') or key == 27:
                    stop_requested = True
                time.sleep(0.03)

            T = base_T_ee_from_move_l_pose(arrived_tcp_pose)

            if USE_CAMERA:
                res = calib.capture(T)
            else:
                res = "robot move only"

        print(f"[AUTO CAPTURE RESULT] {res}")

        res_str = str(res).lower()
        if any(w in res_str for w in ("ok", "success", "saved", "virtual", "robot move only")):
            success_count += 1
            show_sampler_status(
                viewer_window, calib, pose_list, current_index,
                initial_pose_vec, radius_mm,
                status_lines=[f"sample {next_index}: CAPTURE OK"],
                flash_text="CAPTURE OK",
                flash_color=(0, 180, 0),
            )
        else:
            fail_count += 1
            show_sampler_status(
                viewer_window, calib, pose_list, current_index,
                initial_pose_vec, radius_mm,
                status_lines=[f"sample {next_index}: CAPTURE FAIL"],
                flash_text="CAPTURE FAIL",
                flash_color=(0, 0, 180),
            )

        time.sleep(0.4)

        if stop_requested:
            print("[AUTO] stop requested by user")
            break

    print("\n===== AUTO MOVE + CAPTURE DONE =====")
    print(f"success_count = {success_count}")
    print(f"fail_count = {fail_count}")
    if USE_CAMERA and calib is not None:
        print(f"saved samples = {len(calib.collector.samples)}")

    show_sampler_status(
        viewer_window, calib, pose_list, current_index,
        initial_pose_vec, radius_mm,
        status_lines=[
            "AUTO CAPTURE DONE",
            f"success={success_count}, fail={fail_count}",
            "r: calibrate / p: print pose list / q: quit",
        ],
        flash_text="AUTO DONE",
        flash_color=(0, 180, 0),
    )

    return success_count, fail_count


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

if __name__ == "__main__":
    robot = None
    rc = None
    robot_data = None
    calib = None

    try:
        debug("program start")

        # ── 로봇 연결 ──────────────────────────────
        if USE_ROBOT:
            robot = rb.Cobot(ROBOT_IP)
            rc = rb.ResponseCollector()
            robot_data = rb.CobotData(ROBOT_IP)

            robot.set_operation_mode(
                rc,
                rb.OperationMode.Simulation if SIM_MODE else rb.OperationMode.Real
            )
            robot.set_speed_bar(rc, 0.3)
            robot.flush(rc)
            rc.error().throw_if_not_empty()
            debug("robot ready")
        else:
            debug("TEST MODE: robot skipped")

        # ── RealSense 초기화 ───────────────────────
        if USE_CAMERA:
            calib = RealSenseCharucoHandEye(
                width=1280, height=720, fps=30,
                save_dir=str(SAVE_DIR),
            )
            debug("realsense ready")
        else:
            calib = None
            debug("TEST MODE: realsense skipped")

        viewer_window = "RealSense Viewer"
        capture_flash_until = 0.0
        capture_flash_text = ""
        capture_flash_color = (0, 180, 0)

        # 1. 초기 pose 입력
        initial_pose_vec = input_initial_pose_vec()
        debug(f"initial_pose_vec = {initial_pose_vec}")

        # 2. 초기 위치로 이동
        if USE_ROBOT:
            print("\n[STEP 2] 초기 위치로 이동합니다...")
            move_l_blocking(
                robot, rc, robot_data, initial_pose_vec,
                MOVE_SPEED, MOVE_ACCEL
            )
            print("[STEP 2] 이동 완료")

        # 3. 보드까지 거리 측정 → radius 입력
        if not USE_CAMERA:
            radius_mm = RADIUS_MM
            debug(f"TEST MODE radius_mm = {radius_mm}")
        else:
            print("\n[STEP 3] 보드까지 거리 측정 중... (viewer 창 확인)")
            cv2.namedWindow(viewer_window, cv2.WINDOW_NORMAL)

            measured_dist_mm = None
            print("보드가 감지되면 거리가 표시됩니다. 준비되면 SPACE를 누르세요.")

            while True:
                frame = None if calib is None else calib.get_latest_bgr()
                if frame is not None:
                    ok, rvec, tvec, debug_img, msg = calib.collector.detect_board_pose(frame, draw=True)
                    if debug_img is None:
                        debug_img = frame.copy()

                    dist_text = "board: N/A"
                    if ok and tvec is not None:
                        tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(-1)
                        measured_dist_mm = float(np.linalg.norm(tvec_arr) * 1000.0)
                        dist_text = f"board_dist: {measured_dist_mm:.1f} mm"

                    lines = [
                        "STEP 3: 보드까지 거리 측정",
                        f"charuco: {'OK' if ok else 'FAIL'} ({msg})",
                        dist_text,
                        "SPACE: 이 거리로 radius 설정  /  q: 종료",
                    ]
                    show = make_overlay(debug_img, lines)
                    cv2.imshow(viewer_window, show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q') or key == 27:
                    raise SystemExit("user quit at step 3")

            cv2.destroyWindow(viewer_window)

            fallback = measured_dist_mm if measured_dist_mm else RADIUS_MM
            radius_mm = input_radius_mm(fallback)
            debug(f"radius_mm = {radius_mm}")

        # 4. trajectory preview
        pose_list = build_multi_plane_grid_pose_list(
            initial_pose_vec,
            base_dist_mm=radius_mm,
            dist_offsets_mm=GRID_DIST_OFFSETS_MM,
            x_offsets_mm=GRID_U_OFFSETS_MM,
            z_offsets_mm=GRID_V_OFFSETS_MM,
        )
        pose_list, removed_pose_info = filter_unsafe_poses(pose_list)
        debug(f"pose_list built: len={len(pose_list)}")

        show_trajectory_3d(
            pose_list,
            initial_pose_vec,
            radius_mm
        )

        print(f"\n[STEP 4] 궤적 미리보기 — 총 {len(pose_list)}개 포즈")
        print("SPACE: 시작  /  'i': 초기 pose 재입력  /  'q': 종료")

        cv2.namedWindow(viewer_window, cv2.WINDOW_NORMAL)

        while True:
            frame = None if calib is None else calib.get_latest_bgr()
            base_img = frame.copy() if frame is not None else np.zeros((720, 1280, 3), np.uint8)

            show = draw_trajectory_preview(base_img, pose_list, initial_pose_vec, radius_mm)
            cv2.imshow(viewer_window, show)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                print("[STEP 4] 시작!")
                break

            elif key == ord('i'):
                cv2.destroyWindow(viewer_window)
                initial_pose_vec = input_initial_pose_vec()

                if USE_ROBOT:
                    print("\n[재이동] 초기 위치로 이동합니다...")
                    move_l_blocking(
                        robot, rc, robot_data, initial_pose_vec,
                        MOVE_SPEED, MOVE_ACCEL
                    )
                    radius_mm = input_radius_mm(radius_mm)

                pose_list = build_multi_plane_grid_pose_list(
                    initial_pose_vec,
                    base_dist_mm=radius_mm,
                    dist_offsets_mm=GRID_DIST_OFFSETS_MM,
                    x_offsets_mm=GRID_U_OFFSETS_MM,
                    z_offsets_mm=GRID_V_OFFSETS_MM,
                )
                pose_list, removed_pose_info = filter_unsafe_poses(pose_list)
                debug(f"pose_list rebuilt: len={len(pose_list)}")
                cv2.namedWindow(viewer_window, cv2.WINDOW_NORMAL)

            elif key == ord('q') or key == 27:
                raise SystemExit("user quit at step 4")

        # 5. auto calibration / sample loop
        current_index = -1

        if AUTO_MOVE_CAPTURE:
            success_count, fail_count = auto_move_and_capture_all(
                robot=robot,
                rc=rc,
                robot_data=robot_data,
                calib=calib,
                viewer_window=viewer_window,
                pose_list=pose_list,
                initial_pose_vec=initial_pose_vec,
                radius_mm=radius_mm,
            )

            if USE_CAMERA and AUTO_RUN_CALIBRATION_AFTER_CAPTURE:
                res = calib.run_calibration("TSAI")
                print("\n=== AUTO CALIBRATION RESULT ===")
                print(res)

            print("\n===== AUTO SAMPLER FINISHED =====")
            print("r: calibrate / p: 현재 pose list 출력 / q: quit")

            while True:
                frame = None if calib is None else calib.get_latest_bgr()
                if frame is not None:
                    ok, rvec, tvec, debug_img, msg = calib.collector.detect_board_pose(frame, draw=True)
                    if debug_img is None:
                        debug_img = frame.copy()

                    board_dist_text = "board_dist_mm: N/A"
                    if ok and tvec is not None:
                        tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(-1)
                        board_dist_mm_cur = float(np.linalg.norm(tvec_arr) * 1000.0)
                        board_dist_text = f"board_dist_mm: {board_dist_mm_cur:.1f}"

                    num_samples = len(calib.collector.samples)
                    lines = [
                        f"charuco: {'OK' if ok else 'FAIL'} ({msg})",
                        board_dist_text,
                        f"saved samples: {num_samples}",
                        "AUTO SAMPLER FINISHED",
                        "r=calibrate  p=print  q=quit",
                    ]
                    show = make_overlay(debug_img, lines)
                else:
                    show = np.zeros((720, 1280, 3), np.uint8)
                    show = make_overlay(show, [
                        "AUTO SAMPLER FINISHED",
                        "r=calibrate  p=print  q=quit",
                    ])

                show = draw_grid_minimap(
                    show, pose_list, len(pose_list) - 1, initial_pose_vec, radius_mm
                )
                cv2.imshow(viewer_window, show)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('r'):
                    if not USE_CAMERA:
                        print("[TEST] calibration disabled")
                    else:
                        res = calib.run_calibration("TSAI")
                        print("\n=== RESULT ===")
                        print(res)

                elif key == ord('p'):
                    print("\n[pose list]")
                    for i, p in enumerate(pose_list):
                        print(i, p)

                elif key == ord('q') or key == 27:
                    break

        else:
            print("\n===== Hand-in-Arm Plane Grid Sampler =====")
            print(f"num poses = {len(pose_list)}")
            print("AUTO_MOVE_CAPTURE=False 입니다. 기존 수동 모드는 제거하지 않았습니다.")
            print("q: quit")

            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    break

    except SystemExit as e:
        print(f"[EXIT] {e}")
    except Exception as e:
        print("\n[EXCEPTION]")
        print(type(e).__name__, e)
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if calib is not None:
            try:
                calib.stop()
            except Exception as e:
                print("[FINALLY-EXCEPTION] calib.stop()", e)
        debug("program end")