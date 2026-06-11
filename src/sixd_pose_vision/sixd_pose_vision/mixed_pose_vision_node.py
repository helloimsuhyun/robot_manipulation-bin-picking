#!/usr/bin/env python3
"""
Mixed 6D pose vision node.

object mode: RealSense RGB-D -> YOLO-seg preview -> trigger topic -> allowed-id nearest object -> FoundationPose register + short tracking refinement 6D pose
insert mode: RealSense RGB-D -> YOLO-seg -> existing depth PCA + template yaw

Important paths are ROS parameters:
  foundationpose_repo_path, cad_dir, template_dir, object_yolo_path, insert_yolo_path


cd /home/choisuhyun/course/robot_manipulation-bin-picking

rm -rf build/sixd_pose_vision install/sixd_pose_vision

source /opt/ros/humble/setup.bash
conda activate cource

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/choisuhyun/course/robot_manipulation-bin-picking/sFoundationPose:${PYTHONPATH:-}"

python -m colcon build --packages-select sixd_pose_vision
source install/setup.bash

ros2 launch sixd_pose_vision mixed_pose_vision.launch.py
"""

import sys
import json
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import trimesh
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from ultralytics import YOLO


# =============================================================================
# Common geometry utilities
# =============================================================================

def orthonormalize_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt


def build_K(intrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mask_from_polygon(mask_xy: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if mask_xy is not None and len(mask_xy) > 0:
        cv2.fillPoly(mask, [mask_xy.astype(np.int32)], 255)
    return mask


def depth_median_in_mask(depth_image: np.ndarray, mask: np.ndarray, depth_scale: float) -> float:
    z = depth_image[mask > 0].astype(np.float32) * float(depth_scale)
    z = z[z > 0]
    return float(np.median(z)) if len(z) > 0 else float("inf")


def depth_front_median_score_in_mask(
    depth_image: np.ndarray,
    mask: np.ndarray,
    depth_scale: float,
    front_ratio: float = 0.35,
    min_valid_pixels: int = 30,
) -> float:
    """
    Nearest-object selection용 robust depth score.

    전체 mask median(p50)이 아니라, 가까운 쪽 depth subset의 median을 사용한다.

    절차:
      1. mask 내부 valid depth만 수집
      2. invalid depth 0 제거
      3. depth 오름차순 정렬
      4. 가까운 쪽 front_ratio만 선택
      5. 그 subset의 median 반환
    """
    if mask is None:
        return float("inf")

    z = depth_image[mask > 0].astype(np.float32) * float(depth_scale)
    z = z[z > 0]

    if len(z) < min_valid_pixels:
        return float("inf")

    z_sorted = np.sort(z)

    front_count = int(round(len(z_sorted) * float(front_ratio)))
    front_count = max(min_valid_pixels, front_count)
    front_count = min(front_count, len(z_sorted))

    z_front = z_sorted[:front_count]

    return float(np.median(z_front))


def deproject_pixel_to_camera(
    u: float,
    v: float,
    z_m: float,
    intrinsics,
) -> Optional[np.ndarray]:
    """Deproject one image pixel and depth into camera coordinates.

    Returns [x, y, z] in meters in the camera optical frame.
    """
    z_m = float(z_m)
    if not np.isfinite(z_m) or z_m <= 0.0:
        return None

    x = (float(u) - float(intrinsics.ppx)) * z_m / float(intrinsics.fx)
    y = (float(v) - float(intrinsics.ppy)) * z_m / float(intrinsics.fy)
    return np.array([x, y, z_m], dtype=np.float64)


def patch_depth_stats_at_pixel(
    depth_image: np.ndarray,
    u: int,
    v: int,
    depth_scale: float,
    radius_px: int = 5,
) -> Dict:
    """Robust depth stats around an image-plane proposal point.

    Used for empty-space candidates. Depth values are in meters.
    """
    h, w = depth_image.shape[:2]
    r = max(1, int(radius_px))
    x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)

    patch = depth_image[y0:y1, x0:x1].astype(np.float32) * float(depth_scale)
    total_px = int(patch.size)
    valid = patch[patch > 0]

    if total_px <= 0 or len(valid) == 0:
        return {
            "valid": False,
            "median_m": None,
            "valid_ratio": 0.0,
            "spread_p90_p10_m": None,
            "valid_px": 0,
            "total_px": total_px,
        }

    p10 = float(np.percentile(valid, 10))
    p50 = float(np.percentile(valid, 50))
    p90 = float(np.percentile(valid, 90))

    return {
        "valid": True,
        "median_m": p50,
        "valid_ratio": float(len(valid) / max(total_px, 1)),
        "spread_p90_p10_m": float(p90 - p10),
        "valid_px": int(len(valid)),
        "total_px": total_px,
    }


def estimate_mask_translation_camera(
    depth_image: np.ndarray,
    mask_img: np.ndarray,
    intrinsics,
    depth_scale: float,
    z_reject_m: float = 0.05,
) -> Optional[np.ndarray]:
    """Estimate only translation [x,y,z] of a detected object in camera frame.

    This is not a 6D pose. It is a robust median of masked RGB-D points, useful
    for downstream empty-space/world collision filtering before FoundationPose.
    """
    if mask_img is None:
        return None

    rows, cols = np.where(mask_img > 0)
    if len(rows) < 10:
        return None

    z = depth_image[rows, cols].astype(np.float32) * float(depth_scale)
    valid = z > 0
    rows = rows[valid]
    cols = cols[valid]
    z = z[valid]

    if len(z) < 10:
        return None

    z_med = float(np.median(z))
    keep = np.abs(z - z_med) < float(z_reject_m)
    rows = rows[keep]
    cols = cols[keep]
    z = z[keep]

    if len(z) < 10:
        return None

    x = (cols.astype(np.float32) - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y = (rows.astype(np.float32) - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    pts = np.stack([x, y, z], axis=1)

    return np.median(pts, axis=0).astype(np.float64)


def pose_to_dict(pose_mat: np.ndarray, class_name: str, confidence: float, extra: Optional[Dict] = None) -> Dict:
    pose_mat = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
    R = orthonormalize_R(pose_mat[:3, :3])
    t = pose_mat[:3, 3]
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    out = {
        "class": class_name,
        "confidence": round(float(confidence), 3),
        "position": {
            "x": round(float(t[0]), 4),
            "y": round(float(t[1]), 4),
            "z": round(float(t[2]), 4),
        },
        "orientation": {
            "x": round(float(quat[0]), 6),
            "y": round(float(quat[1]), 6),
            "z": round(float(quat[2]), 6),
            "w": round(float(quat[3]), 6),
        },
        "pose_matrix": pose_mat.tolist(),
    }
    if extra:
        out.update(extra)
    return out


def pca_pose_to_dict(
    pose_mat: np.ndarray,
    class_name: str,
    confidence: float,
    yaw_deg: float,
    yaw_score: float,
    yaw_source: str,
) -> Dict:
    pose_mat = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
    R = orthonormalize_R(pose_mat[:3, :3])
    t = pose_mat[:3, 3]
    return {
        "class": class_name,
        "confidence": round(float(confidence), 3),
        "position": {
            "x": round(float(t[0]), 4),
            "y": round(float(t[1]), 4),
            "z": round(float(t[2]), 4),
        },
        "orientation": {
            "axis_x": [round(float(v), 4) for v in R[:, 0]],
            "axis_y": [round(float(v), 4) for v in R[:, 1]],
            "axis_z": [round(float(v), 4) for v in R[:, 2]],
        },
        "yaw_deg": round(float(yaw_deg), 2),
        "yaw_score": round(float(yaw_score), 3),
        "yaw_source": yaw_source,
        "pose_matrix": pose_mat.tolist(),
    }


def make_pose_stamped(pose_mat: np.ndarray, frame_id: str) -> PoseStamped:
    pose_mat = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
    R = orthonormalize_R(pose_mat[:3, :3])
    t = pose_mat[:3, 3]
    q = Rotation.from_matrix(R).as_quat()
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def draw_pose_axis(image: np.ndarray, pose_mat: np.ndarray, K: np.ndarray, axis_len: float = 0.10) -> None:
    pose_mat = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4)
    R = pose_mat[:3, :3]
    t = pose_mat[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def proj(pt3d):
        x, y, z = pt3d
        if z <= 0:
            return None
        return (int(x * fx / z + cx), int(y * fy / z + cy))

    origin = proj(t)
    if origin is None:
        return
    for axis_vec, color in [(R[:, 0], (0, 0, 255)), (R[:, 1], (0, 255, 0)), (R[:, 2], (255, 0, 0))]:
        p = proj(t + axis_vec * axis_len)
        if p is not None:
            cv2.arrowedLine(image, origin, p, color, 2, tipLength=0.2)


def draw_projected_3d_bbox(
    image: np.ndarray,
    ob_in_cam: np.ndarray,
    K: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Project and draw a CAD 3D bounding box.

    bbox format:
      [[xmin, ymin, zmin],
       [xmax, ymax, zmax]]
    ob_in_cam:
      4x4 transform from object/bbox coordinate to camera coordinate.
    """
    ob_in_cam = np.asarray(ob_in_cam, dtype=np.float64).reshape(4, 4)
    bbox = np.asarray(bbox, dtype=np.float64).reshape(2, 3)

    xmin, ymin, zmin = bbox[0]
    xmax, ymax, zmax = bbox[1]

    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )

    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float64)], axis=1)
    pts_cam = (ob_in_cam @ corners_h.T).T[:, :3]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pts_2d = []
    for x, y, z in pts_cam:
        if z <= 0:
            pts_2d.append(None)
        else:
            pts_2d.append((int(x * fx / z + cx), int(y * fy / z + cy)))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for a, b in edges:
        if pts_2d[a] is not None and pts_2d[b] is not None:
            cv2.line(image, pts_2d[a], pts_2d[b], color, thickness, cv2.LINE_AA)


# =============================================================================
# Insert mode: existing template yaw + PCA pose
# =============================================================================

def normalize_binary(img: Optional[np.ndarray], match_size: int) -> Optional[np.ndarray]:
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (match_size, match_size), interpolation=cv2.INTER_NEAREST)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img.astype(np.uint8)


def rotate_keep_size(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def crop_mask_to_square(mask_img: np.ndarray, pad_ratio: float = 0.2) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_img > 0)
    if len(xs) < 10:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    side = max(x1 - x0 + 1, y1 - y0 + 1)
    side += 2 * int(round(side * pad_ratio))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
    out = np.zeros((side, side), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(mask_img.shape[1], x1), min(mask_img.shape[0], y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    out[dy0:dy1, dx0:dx1] = mask_img[sy0:sy1, sx0:sx1]
    return out


def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    a_bin, b_bin = a > 0, b > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union > 0 else 0.0



def depth_stats_in_mask(
    depth_image: np.ndarray,
    mask_img: np.ndarray,
    depth_scale: float,
) -> Dict:
    """Return detailed depth statistics inside a binary mask.

    Used only for debugging. All depth values are in meters.
    """
    if mask_img is None:
        return {
            "area_px": 0,
            "valid_px": 0,
            "valid_ratio": 0.0,
            "min": None,
            "p05": None,
            "p10": None,
            "p20": None,
            "p35": None,
            "p50": None,
            "p75": None,
            "max": None,
            "spread_p50_p10": None,
            "spread_p50_p20": None,
        }

    z = depth_image[mask_img > 0].astype(np.float32) * float(depth_scale)
    valid_z = z[z > 0]

    area_px = int(np.sum(mask_img > 0))
    valid_px = int(len(valid_z))
    valid_ratio = float(valid_px / area_px) if area_px > 0 else 0.0

    if valid_px == 0:
        return {
            "area_px": area_px,
            "valid_px": valid_px,
            "valid_ratio": round(valid_ratio, 4),
            "min": None,
            "p05": None,
            "p10": None,
            "p20": None,
            "p35": None,
            "p50": None,
            "p75": None,
            "max": None,
            "spread_p50_p10": None,
            "spread_p50_p20": None,
        }

    p05 = float(np.percentile(valid_z, 5))
    p10 = float(np.percentile(valid_z, 10))
    p20 = float(np.percentile(valid_z, 20))
    p35 = float(np.percentile(valid_z, 35))
    p50 = float(np.percentile(valid_z, 50))
    p75 = float(np.percentile(valid_z, 75))

    return {
        "area_px": area_px,
        "valid_px": valid_px,
        "valid_ratio": round(valid_ratio, 4),
        "min": round(float(np.min(valid_z)), 4),
        "p05": round(p05, 4),
        "p10": round(p10, 4),
        "p20": round(p20, 4),
        "p35": round(p35, 4),
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "max": round(float(np.max(valid_z)), 4),
        "spread_p50_p10": round(float(p50 - p10), 4),
        "spread_p50_p20": round(float(p50 - p20), 4),
    }


def save_mask_point_cloud_ply(
    path: Path,
    depth_image: np.ndarray,
    mask_img: np.ndarray,
    color_image: np.ndarray,
    intrinsics,
    depth_scale: float,
    max_points: int = 30000,
) -> int:
    """Save masked RGB-D points as an ASCII PLY file.

    Returns the number of saved points. Coordinates are in the camera frame, meters.
    """
    if mask_img is None:
        return 0

    rows, cols = np.where(mask_img > 0)
    if len(rows) == 0:
        return 0

    z = depth_image[rows, cols].astype(np.float32) * float(depth_scale)
    valid = z > 0
    rows = rows[valid]
    cols = cols[valid]
    z = z[valid]

    if len(z) == 0:
        return 0

    x = (cols.astype(np.float32) - intrinsics.ppx) * z / intrinsics.fx
    y = (rows.astype(np.float32) - intrinsics.ppy) * z / intrinsics.fy
    pts = np.stack([x, y, z], axis=1)
    bgr = color_image[rows, cols].astype(np.uint8)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        bgr = bgr[idx]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(pts, bgr):
            # color_image is BGR, PLY expects RGB.
            b, g, r = int(c[0]), int(c[1]), int(c[2])
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")

    return int(len(pts))


def get_point_cloud(depth_image: np.ndarray, mask_img: np.ndarray, intrinsics, depth_scale: float) -> Optional[np.ndarray]:
    rows, cols = np.where(mask_img > 0)
    z_vals = depth_image[rows, cols].astype(np.float32) * float(depth_scale)
    valid = z_vals > 0
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
    if len(z_vals) < 10:
        return None
    z_med = np.median(z_vals)
    valid = np.abs(z_vals - z_med) < 0.05
    rows, cols, z_vals = rows[valid], cols[valid], z_vals[valid]
    if len(z_vals) < 10:
        return None
    x = (cols - intrinsics.ppx) * z_vals / intrinsics.fx
    y = (rows - intrinsics.ppy) * z_vals / intrinsics.fy
    return np.stack([x, y, z_vals], axis=1)


def estimate_pca_pose(points: np.ndarray) -> np.ndarray:
    centroid = np.median(points, axis=0)
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = orthonormalize_R(eigvecs[:, order])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = centroid
    return T


# =============================================================================
# Object mode: FoundationPose wrapper
# =============================================================================

class FPEstimator:
    def __init__(self, class_name, mesh, foundationpose_cls, glctx, scorer, refiner, debug=0, debug_dir=""):
        self.class_name = class_name
        kwargs = dict(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=int(debug),
        )
        if debug_dir:
            kwargs["debug_dir"] = debug_dir
        self.estimator = foundationpose_cls(**kwargs)
        self.registered = False
        self.last_score = 1.0

    def estimate(
        self,
        rgb_bgr,
        depth_uint16,
        mask_uint8,
        K,
        register_iter,
        track_iter,
        track_loss_thr,
        use_tracking,
        depth_scale: float = 0.001,
    ):
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        depth_scale = float(depth_scale)
        if depth_scale <= 0.0:
            raise ValueError(f"Invalid depth_scale={depth_scale}. It must be positive.")

        depth = depth_uint16.astype(np.float32) * depth_scale
        need_register = (not use_tracking) or (not self.registered) or (self.last_score < track_loss_thr)
        if need_register:
            pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=(mask_uint8 > 0), iteration=int(register_iter))
            self.registered = True
        else:
            pose = self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=int(track_iter))
        if hasattr(self.estimator, "last_score"):
            try:
                self.last_score = float(self.estimator.last_score)
            except Exception:
                pass
        return np.asarray(pose, dtype=np.float64).reshape(4, 4)

    def reset(self):
        self.registered = False
        self.last_score = 1.0


# =============================================================================
# ROS2 node
# =============================================================================

class MixedPoseVisionNode(Node):
    def __init__(self):
        super().__init__("mixed_pose_vision_node")

        # Path parameters
        self.declare_parameter("foundationpose_repo_path", str(Path.home() / "FoundationPose"))
        self.declare_parameter("cad_dir", "")
        self.declare_parameter("template_dir", "")
        self.declare_parameter("object_yolo_path", "")
        self.declare_parameter("insert_yolo_path", "")
        self.declare_parameter("mesh_scale", 0.001)

        # Topic parameters
        self.declare_parameter("object_topic", "/object_poses")
        self.declare_parameter("insert_topic", "/insert_poses")
        self.declare_parameter("object_pose_topic", "/object_pose_stamped")
        self.declare_parameter("insert_pose_topic", "/insert_pose_stamped")
        self.declare_parameter("detect_mode_topic", "/detect_mode")
        self.declare_parameter("object_trigger_topic", "/manipulation/object_6d_trigger")
        self.declare_parameter("empty_space_topic", "/empty_space_candidates")

        # Empty-space proposal parameters
        self.declare_parameter("empty_space_enable", True)
        self.declare_parameter("empty_grid_step_px", 40)
        self.declare_parameter("empty_max_candidates", 30)
        self.declare_parameter("empty_roi_x_min", 70)
        self.declare_parameter("empty_roi_y_min", 60)
        self.declare_parameter("empty_roi_x_max", -1)  # -1 means image_width - margin
        self.declare_parameter("empty_roi_y_max", -1)  # -1 means image_height - margin
        self.declare_parameter("empty_roi_right_margin", 70)
        self.declare_parameter("empty_roi_bottom_margin", 50)
        self.declare_parameter("empty_mask_dilate_px", 22)
        self.declare_parameter("empty_depth_patch_radius_px", 6)
        self.declare_parameter("empty_depth_valid_ratio_min", 0.50)
        self.declare_parameter("empty_depth_spread_max_m", 0.030)
        self.declare_parameter("empty_space_vis_hold_sec", 3.0)

        # Runtime parameters
        self.declare_parameter("default_mode", "object")
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("conf_thresh", 0.4)
        # 템플릿 매칭
        self.declare_parameter("angle_step_deg", 1)
        self.declare_parameter("match_size", 160)
        
        # 리얼센스
        self.declare_parameter("color_width", 848)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("fps", 30)

        # FoundationPose parameters
        self.declare_parameter("fp_register_iter", 5)
        self.declare_parameter("fp_track_iter", 2)
        self.declare_parameter("fp_track_loss_thr", 0.2)

        # Trigger one-shot refinement:
        #   0 -> old behavior: register-only and publish immediately.
        #   N -> fresh register once, then track_one for N additional frames, publish final pose, reset.
        self.declare_parameter("fp_trigger_track_frames", 10)
        self.declare_parameter("fp_trigger_track_use_new_frames", True)
        self.declare_parameter("fp_debug", 0)
        self.declare_parameter("fp_debug_dir", "/home/choisuhyun/course/robot_manipulation-bin-picking/FoundationPose/debug_ros")

        # Priority/depth debug artifacts.
        # priority_debug_save=True  -> save debug files every 6D trigger.
        # priority_debug_save=False -> do not save debug files.
        # priority_debug_dir is relative to the current execution directory by default.
        self.declare_parameter("priority_debug_save", True)
        self.declare_parameter("priority_debug_dir", "debug_priority")

        # Priority tie-break parameters.
        # 기본 우선순위는 depth_score가 작은 물체, 즉 카메라에 더 가까운 물체입니다.
        # 단, 카메라 좌표계 XY 위치가 비슷하고 depth_score 차이도 priority_depth_tie_m 이하이면
        # tie group으로 묶습니다. 그 안에서 hole confidence가 비교 후보보다 높으면 hole을 우선합니다.
        self.declare_parameter("priority_depth_tie_m", 0.010)
        self.declare_parameter("priority_xy_tie_m", 0.040)
        self.declare_parameter("priority_hole_class", "hole")
        self.declare_parameter("priority_hole_conf_margin", 0.03)

        self.foundationpose_repo_path = Path(str(self.get_parameter("foundationpose_repo_path").value)).expanduser()
        self.cad_dir = Path(str(self.get_parameter("cad_dir").value)).expanduser()
        self.template_dir = Path(str(self.get_parameter("template_dir").value)).expanduser()
        self.object_yolo_path = Path(str(self.get_parameter("object_yolo_path").value)).expanduser()
        self.insert_yolo_path = Path(str(self.get_parameter("insert_yolo_path").value)).expanduser()
        self.mesh_scale = float(self.get_parameter("mesh_scale").value)
        self.conf_thresh = float(self.get_parameter("conf_thresh").value)
        self.angle_step_deg = max(1, int(self.get_parameter("angle_step_deg").value))
        self.match_size = int(self.get_parameter("match_size").value)
        self.enable_visualization = bool(self.get_parameter("enable_visualization").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.fp_debug = int(self.get_parameter("fp_debug").value)
        self.fp_debug_dir = str(self.get_parameter("fp_debug_dir").value).strip()
        if not self.fp_debug_dir:
            self.fp_debug_dir = str(self.foundationpose_repo_path / "debug_ros")

        Path(self.fp_debug_dir).mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"[FP] debug_dir: {self.fp_debug_dir}")

        self.priority_debug_save = bool(self.get_parameter("priority_debug_save").value)
        priority_debug_dir_param = Path(str(self.get_parameter("priority_debug_dir").value)).expanduser()
        if priority_debug_dir_param.is_absolute():
            self.priority_debug_dir = priority_debug_dir_param
        else:
            self.priority_debug_dir = Path.cwd() / priority_debug_dir_param
        self.priority_debug_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"[PRIORITY_DEBUG] save={self.priority_debug_save} dir={self.priority_debug_dir}"
        )

        self.priority_depth_tie_m = float(self.get_parameter("priority_depth_tie_m").value)
        self.priority_xy_tie_m = float(self.get_parameter("priority_xy_tie_m").value)
        self.priority_hole_class = str(self.get_parameter("priority_hole_class").value).strip()
        self.priority_hole_conf_margin = float(self.get_parameter("priority_hole_conf_margin").value)
        self.get_logger().info(
            f"[PRIORITY] rule=nearest_depth_score_with_camera_xy_depth_hole_conf_compare_tie_break "
            f"depth_tie_m={self.priority_depth_tie_m:.3f} "
            f"xy_tie_m={self.priority_xy_tie_m:.3f} "
            f"hole_class={self.priority_hole_class} "
            f"hole_conf_margin={self.priority_hole_conf_margin:.3f}"
        )

        self.mesh_alias = {"cross_insert": "cross", "cylinder_insert": "cylinder", "hole_insert": "hole"}
        self.class_to_id = {
            "cylinder": 0,
            "cylinder_insert": 0,
            "hole": 1,
            "hole_insert": 1,
            "cross": 2,
            "cross_insert": 2,
        }
        self.class_colors = {
            "cross": (0, 220, 0), "cylinder": (0, 140, 255), "hole": (220, 0, 220),
            "cross_insert": (0, 220, 0), "cylinder_insert": (0, 140, 255), "hole_insert": (220, 0, 220),
        }
        self.template_files = {
            "cross": "cross_top.png", "cylinder": "circle_top.png", "hole": "square_top.png",
            "cross_insert": "cross_insert_top.png", "cylinder_insert": "circle_insert_top.png", "hole_insert": "square_insert_top.png",
        }
        self.no_yaw_classes = {"cylinder", "cylinder_insert"}
        self.rotated_templates = {}
        self._fp_estimators = {}
        self._mesh_vis_info = {}

        # Last successful FoundationPose result for persistent object visualization.
        self.last_object_pose_mat = None
        self.last_object_class_name = None
        self.last_object_status_text = ""

        # Last empty-space candidates for visualization hold after a trigger.
        self.last_empty_space_debug = None
        self.last_empty_space_debug_stamp_sec = None

        # Publishers/subscribers
        self.object_pub = self.create_publisher(String, str(self.get_parameter("object_topic").value), 10)
        self.insert_pub = self.create_publisher(String, str(self.get_parameter("insert_topic").value), 10)
        self.empty_space_pub = self.create_publisher(String, str(self.get_parameter("empty_space_topic").value), 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("object_pose_topic").value), 10)
        self.insert_pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("insert_pose_topic").value), 10)
        self.create_subscription(String, str(self.get_parameter("detect_mode_topic").value), self._mode_callback, 10)
        self.create_subscription(String, str(self.get_parameter("object_trigger_topic").value), self._object_trigger_callback, 10)

        # Object 6D is computed only when a trigger message arrives.
        # Trigger payload examples:
        #   "" or "nearest"                       -> nearest detected object
        #   "cross"                               -> nearest detected cross only
        #   '{"class":"cylinder"}'              -> nearest detected cylinder only
        #   '{"allowed_ids":[0,2]}'              -> nearest detected cylinder/cross only
        #   '{"allowed_classes":["cylinder"]}' -> nearest detected cylinder only
        self._object_trigger_pending = False
        self._object_trigger_class: Optional[str] = None
        self._object_trigger_allowed_ids: Optional[List[int]] = None
        self._object_trigger_seq = 0

        self.object_model = self._load_yolo(self.object_yolo_path, "object")
        self.insert_model = self._load_yolo(self.insert_yolo_path, "insert")

        self.mode = str(self.get_parameter("default_mode").value).strip().lower()
        if self.mode not in {"object", "insert"}:
            self.mode = "object"

        self.fp_available = False
        self.FoundationPose = None
        self.glctx = None
        self.scorer = None
        self.refiner = None
        self._init_foundationpose()
        self._load_rotated_templates()

        # RealSense direct capture
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        width = int(self.get_parameter("color_width").value)
        height = int(self.get_parameter("color_height").value)
        fps = int(self.get_parameter("fps").value)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline.start(cfg)
        profile = self.pipeline.get_active_profile()
        self.intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
        self.K = build_K(self.intrinsics)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.align = rs.align(rs.stream.color)

        self.timer = self.create_timer(0.1, self._timer_callback)
        self.get_logger().info(
            f"MixedPoseVisionNode ready | mode={self.mode} | object=triggered FoundationPose register+short-track | insert=PCA+template | "
            f"depth_scale={self.depth_scale:.6f}"
        )

    def _load_yolo(self, path: Path, name: str):
        if not str(path) or str(path) == ".":
            self.get_logger().warn(f"[YOLO] {name}_yolo_path is empty")
            return None
        if not path.exists():
            self.get_logger().warn(f"[YOLO] {name} model not found: {path}")
            return None
        self.get_logger().info(f"[YOLO] {name} model loaded: {path}")
        return YOLO(str(path))

    def _init_foundationpose(self):
        if not self.foundationpose_repo_path.exists():
            self.get_logger().warn(f"[FP] repo path not found: {self.foundationpose_repo_path}")
            return
        sys.path.insert(0, str(self.foundationpose_repo_path))
        try:
            from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
            import nvdiffrast.torch as dr
            self.FoundationPose = FoundationPose
            self.glctx = dr.RasterizeCudaContext()
            self.scorer = ScorePredictor()
            self.refiner = PoseRefinePredictor()
            self.fp_available = True
            self.get_logger().info("[FP] initialized. object mode uses FoundationPose.")
        except Exception as e:
            self.fp_available = False
            self.get_logger().warn(f"[FP] initialization failed: {e}")
            traceback.print_exc()

    def _load_rotated_templates(self):
        self.rotated_templates.clear()
        if not self.template_dir.exists():
            self.get_logger().warn(f"[TEMPLATE] template_dir not found: {self.template_dir}")
            return
        for cls, fname in self.template_files.items():
            path = self.template_dir / fname
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().warn(f"[TEMPLATE] missing: {path}")
                continue
            img = normalize_binary(img, self.match_size)
            if img is None:
                continue
            self.rotated_templates[cls] = []
            for angle in range(0, 180, self.angle_step_deg):
                self.rotated_templates[cls].append((float(angle), rotate_keep_size(img, float(angle))))
            self.get_logger().info(f"[TEMPLATE] loaded: {cls} from {path}")

    def _load_mesh(self, class_name: str) -> trimesh.Trimesh:
        if not self.cad_dir.exists():
            raise FileNotFoundError(f"cad_dir not found: {self.cad_dir}")
        base = self.mesh_alias.get(class_name, class_name)
        candidates = [self.cad_dir / f"{base}{ext}" for ext in [".stl", ".obj", ".ply"]]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError("mesh not found: " + ", ".join(str(p) for p in candidates))
        mesh = trimesh.load(str(path), force="mesh")
        if self.mesh_scale != 1.0:
            mesh.apply_scale(self.mesh_scale)
        return mesh

    def _get_fp_estimator(self, class_name: str) -> Optional[FPEstimator]:
        if class_name in self._fp_estimators:
            return self._fp_estimators[class_name]
        if not self.fp_available:
            return None
        mesh = self._load_mesh(class_name)

        # Precompute CAD oriented bbox visualization data.
        # FoundationPose pose is object_in_camera for the original mesh frame.
        # For the oriented bounds, use center_pose = pose @ inv(to_origin), same as the official demo.
        if class_name not in self._mesh_vis_info:
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
            self._mesh_vis_info[class_name] = {
                "to_origin": to_origin,
                "bbox": bbox,
            }

        est = FPEstimator(
            class_name=class_name,
            mesh=mesh,
            foundationpose_cls=self.FoundationPose,
            glctx=self.glctx,
            scorer=self.scorer,
            refiner=self.refiner,
            debug=self.fp_debug,
            debug_dir=self.fp_debug_dir,
        )
        self._fp_estimators[class_name] = est
        self.get_logger().info(f"[FP] estimator created: {class_name}")
        return est

    def _mode_callback(self, msg: String):
        mode = msg.data.strip().lower()
        if mode not in {"object", "insert"}:
            self.get_logger().warn(f"unknown mode: {mode}")
            return
        if mode == "object" and self.object_model is None:
            self.get_logger().warn("object YOLO model not loaded")
            return
        if mode == "insert" and self.insert_model is None:
            self.get_logger().warn("insert YOLO model not loaded")
            return
        self.mode = mode
        if mode == "object":
            for est in self._fp_estimators.values():
                est.reset()
        self.get_logger().info(f"mode switched -> {mode}")

    def _object_trigger_callback(self, msg: String):
        payload = msg.data.strip()

        target_class = None
        allowed_ids = None

        if payload:
            try:
                data = json.loads(payload)

                if isinstance(data, dict):
                    if "allowed_ids" in data:
                        allowed_ids = []
                        for v in data.get("allowed_ids", []):
                            try:
                                obj_id = int(round(float(v)))
                            except Exception:
                                continue
                            if obj_id in {0, 1, 2} and obj_id not in allowed_ids:
                                allowed_ids.append(obj_id)

                    elif "allowed_classes" in data:
                        allowed_ids = []
                        for cls in data.get("allowed_classes", []):
                            cls = str(cls).strip()
                            if cls in self.class_to_id:
                                obj_id = int(self.class_to_id[cls])
                                if obj_id not in allowed_ids:
                                    allowed_ids.append(obj_id)

                    else:
                        target_class = data.get("class") or data.get("target_class")

            except Exception:
                # Backward compatibility:
                # raw class name, "nearest", etc.
                if payload.lower() not in {"nearest", "object", "trigger", "capture", "1", "true"}:
                    target_class = payload

        if allowed_ids is not None and len(allowed_ids) == 0:
            allowed_ids = None

        if target_class is not None:
            target_class = str(target_class).strip()
            if not target_class:
                target_class = None

        self._object_trigger_pending = True
        self._object_trigger_class = target_class
        self._object_trigger_allowed_ids = allowed_ids
        self._object_trigger_seq += 1
        self.get_logger().info(
            f"[OBJECT_TRIGGER] seq={self._object_trigger_seq} "
            f"target_class={target_class or 'nearest'} "
            f"allowed_ids={allowed_ids if allowed_ids is not None else 'all'}"
        )

    def _consume_object_trigger(self) -> Tuple[bool, Optional[str], Optional[List[int]], int]:
        if not self._object_trigger_pending:
            return False, None, None, self._object_trigger_seq

        target_class = self._object_trigger_class
        allowed_ids = self._object_trigger_allowed_ids
        seq = self._object_trigger_seq

        self._object_trigger_pending = False
        self._object_trigger_class = None
        self._object_trigger_allowed_ids = None

        return True, target_class, allowed_ids, seq

    def _read_aligned_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def _timer_callback(self):
        color, depth = self._read_aligned_rgbd()
        if color is None or depth is None:
            return
        display = color.copy()

        if self.mode == "object":
            triggered, target_class, allowed_ids, trigger_seq = self._consume_object_trigger()
            objects, status = self._process_object_mode(
                color=color,
                depth=depth,
                display=display,
                run_fp=triggered,
                requested_class=target_class,
                requested_allowed_ids=allowed_ids,
                trigger_seq=trigger_seq,
            )
            # Object 6D response is published only for an explicit trigger.
            if triggered:
                self._publish_json(self.object_pub, "object", objects, extra=status)
        else:
            objects = self._process_insert_mode(color, depth, display)
            self._publish_json(self.insert_pub, "insert", objects)

        if self.enable_visualization:
            self._draw_last_empty_space_debug(display)
            cv2.imshow("6D Pose Vision | object=FP, insert=PCA+template", display)
            cv2.waitKey(1)

    def _collect_detections(self, model, color, depth) -> List[Dict]:
        if model is None:
            return []
        results = model(color, conf=self.conf_thresh, verbose=False)[0]
        detections = []
        if results.masks is None:
            return detections
        for i, mask_xy in enumerate(results.masks.xy):
            mask_img = mask_from_polygon(mask_xy, color.shape)
            cls_id = int(results.boxes.cls[i])

            # depth_med: 기존 전체 mask median. 디버그/로그용으로 유지.
            # depth_score: nearest 선택용. 가까운 쪽 35% depth subset의 median.
            depth_med = depth_median_in_mask(depth, mask_img, self.depth_scale)
            depth_score = depth_front_median_score_in_mask(
                depth,
                mask_img,
                self.depth_scale,
                front_ratio=0.35,
                min_valid_pixels=30,
            )

            detections.append({
                "class_name": results.names[cls_id],
                "confidence": float(results.boxes.conf[i]),
                "mask_xy": mask_xy,
                "mask_img": mask_img,
                "depth_med": depth_med,
                "depth_score": depth_score,
                "bbox": results.boxes.xyxy[i].cpu().numpy().astype(int),
            })

        # allowed-id nearest 선택과 preview 대표 detection 모두 priority rule 기준으로 정렬.
        detections = self._sort_detections_with_square_tie_break(detections)
        return detections

    def _depth_value_for_priority(self, det: Dict) -> float:
        return float(det.get("depth_score", det.get("depth_med", float("inf"))))

    def _mask_center_for_priority(self, det: Dict) -> Tuple[float, float]:
        """Return mask centroid in image pixel coordinates.

        Falls back to bbox center if mask moments are invalid.
        """
        mask_img = det.get("mask_img")
        if mask_img is not None:
            m = cv2.moments((mask_img > 0).astype(np.uint8))
            if abs(float(m.get("m00", 0.0))) > 1e-6:
                return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        return 0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2))

    def _mask_center_distance_px(self, a: Dict, b: Dict) -> float:
        ax, ay = self._mask_center_for_priority(a)
        bx, by = self._mask_center_for_priority(b)
        return float(np.hypot(ax - bx, ay - by))

    def _camera_xy_center_for_priority(self, det: Dict) -> Optional[np.ndarray]:
        """Return mask center as camera-frame XY position in meters.

        The image mask center is deprojected using the detection's priority depth.
        This is only for priority grouping, not for final 6D pose estimation.
        """
        u, v = self._mask_center_for_priority(det)
        z = self._depth_value_for_priority(det)

        p_cam = deproject_pixel_to_camera(
            u=float(u),
            v=float(v),
            z_m=float(z),
            intrinsics=self.intrinsics,
        )

        if p_cam is None:
            return None

        return p_cam[:2].astype(np.float64)

    def _camera_xy_distance_for_priority_m(self, a: Dict, b: Dict) -> float:
        """Distance between two detection mask centers in camera-frame XY meters."""
        pa = self._camera_xy_center_for_priority(a)
        pb = self._camera_xy_center_for_priority(b)

        if pa is None or pb is None:
            return float("inf")

        return float(np.linalg.norm(pa - pb))

    def _sort_detections_with_square_tie_break(self, detections: List[Dict]) -> List[Dict]:
        """Sort detections with camera-XY + depth + hole-confidence comparison.

        Priority rule:
          1) Start from the nearest depth_score candidate as the anchor.
          2) Candidates whose camera-frame XY centers are close to the anchor
             become comparison targets.
          3) Inside that local camera-XY group, if depth_score difference is small,
             apply tie-break.
          4) In the tie group, prefer hole only when the best hole confidence is
             higher than the best non-hole confidence by priority_hole_conf_margin.
          5) Otherwise keep normal nearest-depth priority.
        """
        if not detections:
            return detections

        depth_tie_m = max(0.0, float(getattr(self, "priority_depth_tie_m", 0.010)))
        xy_tie_m = max(0.0, float(getattr(self, "priority_xy_tie_m", 0.040)))
        hole_cls = str(getattr(self, "priority_hole_class", "hole"))
        hole_conf_margin = max(0.0, float(getattr(self, "priority_hole_conf_margin", 0.03)))

        # 기본 anchor는 전체 후보 중 가장 가까운 depth_score 후보.
        depth_sorted = sorted(detections, key=lambda d: self._depth_value_for_priority(d))
        anchor = depth_sorted[0]
        anchor_depth = self._depth_value_for_priority(anchor)

        local_xy_group = []
        nonlocal_group = []

        for det in depth_sorted:
            xy_dist_m = self._camera_xy_distance_for_priority_m(det, anchor)
            px_dist = self._mask_center_distance_px(det, anchor)
            depth_diff = abs(self._depth_value_for_priority(det) - anchor_depth)

            det["priority_depth_diff_from_nearest_m"] = float(depth_diff)
            det["priority_xy_dist_from_nearest_m"] = float(xy_dist_m)
            det["priority_center_dist_from_nearest_px"] = float(px_dist)  # debug/backward compatibility
            det["priority_xy_gate"] = bool(xy_dist_m <= xy_tie_m)
            det["priority_position_gate"] = bool(det["priority_xy_gate"])  # debug/backward compatibility
            det["priority_depth_gate"] = bool(depth_diff <= depth_tie_m)
            det["priority_hole_conf_best"] = None
            det["priority_non_hole_conf_best"] = None
            det["priority_hole_conf_margin"] = float(hole_conf_margin)
            det["priority_hole_conf_win"] = False
            det["priority_hole_tie_applied"] = False
            det["priority_square_tie_applied"] = False  # debug/backward compatibility

            if det["priority_xy_gate"]:
                local_xy_group.append(det)
            else:
                nonlocal_group.append(det)

        # 카메라 XY 위치가 비슷한 후보 안에서만 depth tie를 판단.
        local_depth_tie_group = []
        local_depth_other_group = []

        for det in local_xy_group:
            if det["priority_depth_gate"]:
                local_depth_tie_group.append(det)
            else:
                local_depth_other_group.append(det)

        # XY도 비슷하고 depth도 비슷할 때만 confidence 비교 기반 hole tie-break.
        if len(local_depth_tie_group) >= 2:
            holes = [
                det for det in local_depth_tie_group
                if str(det.get("class_name", "")) == hole_cls
            ]
            non_holes = [
                det for det in local_depth_tie_group
                if str(det.get("class_name", "")) != hole_cls
            ]

            best_hole_conf = max(
                [float(det.get("confidence", 0.0)) for det in holes],
                default=-1.0,
            )
            best_non_hole_conf = max(
                [float(det.get("confidence", 0.0)) for det in non_holes],
                default=-1.0,
            )

            hole_conf_wins = (
                len(holes) > 0
                and (
                    len(non_holes) == 0
                    or best_hole_conf >= best_non_hole_conf + hole_conf_margin
                )
            )

            for det in local_depth_tie_group:
                cls = str(det.get("class_name", ""))
                is_hole = cls == hole_cls
                det["priority_hole_conf_best"] = float(best_hole_conf)
                det["priority_non_hole_conf_best"] = float(best_non_hole_conf)
                det["priority_hole_conf_win"] = bool(hole_conf_wins)
                det["priority_hole_tie_applied"] = bool(hole_conf_wins and is_hole)
                det["priority_square_tie_applied"] = bool(hole_conf_wins and is_hole)

            if hole_conf_wins:
                local_depth_tie_group.sort(
                    key=lambda det: (
                        0 if str(det.get("class_name", "")) == hole_cls else 1,
                        -float(det.get("confidence", 0.0)),
                        self._depth_value_for_priority(det),
                    )
                )
            else:
                # hole confidence가 비교 후보보다 충분히 높지 않으면 depth 우선.
                local_depth_tie_group.sort(
                    key=lambda det: self._depth_value_for_priority(det)
                )
        else:
            local_depth_tie_group.sort(
                key=lambda det: self._depth_value_for_priority(det)
            )

        # 같은 XY 그룹이지만 depth 차이가 확실하면 depth 우선.
        local_depth_other_group.sort(
            key=lambda det: self._depth_value_for_priority(det)
        )

        # XY 위치가 다른 후보도 depth 우선.
        nonlocal_group.sort(
            key=lambda det: self._depth_value_for_priority(det)
        )

        return local_depth_tie_group + local_depth_other_group + nonlocal_group

    def _available_object_info(self, detections: List[Dict]) -> Tuple[List[int], List[str]]:
        """Return unique currently detected object ids/classes in nearest-depth order."""
        available_ids = []
        available_classes = []

        for det in detections:
            cls = str(det.get("class_name", ""))
            if cls not in self.class_to_id:
                continue

            obj_id = int(self.class_to_id[cls])
            canonical_cls = self.mesh_alias.get(cls, cls)

            if obj_id not in available_ids:
                available_ids.append(obj_id)
                available_classes.append(canonical_cls)

        return available_ids, available_classes

    def _make_priority_table(self, detections: List[Dict]) -> List[Dict]:
        """Build a JSON-friendly priority table from already-sorted detections.

        Current rule:
          smaller depth_score_m -> closer to camera -> higher priority.
          if camera XY distance and depth_score difference are small, hole is preferred
          only when its YOLO confidence is higher than comparison candidates.

        depth_score is not the full mask median. It is the median of the nearest
        front-side depth subset computed in depth_front_median_score_in_mask().
        """
        table = []

        for rank, det in enumerate(detections, start=1):
            cls = str(det.get("class_name", "unknown"))
            obj_id = int(self.class_to_id.get(cls, -1))

            depth_score = float(det.get("depth_score", det.get("depth_med", float("inf"))))
            depth_med = float(det.get("depth_med", float("inf")))
            conf = float(det.get("confidence", 0.0))

            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            area_px = int(max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1)))

            table.append({
                "rank": int(rank),
                "selected": bool(rank == 1),
                "class_name": cls,
                "class_id": obj_id,
                "confidence": round(conf, 3),
                "depth_score_m": round(depth_score, 4),
                "depth_median_m": round(depth_med, 4),
                "bbox_area_px": area_px,
                "priority_depth_diff_from_nearest_m": round(
                    float(det.get("priority_depth_diff_from_nearest_m", 0.0)), 4
                ),
                "priority_xy_dist_from_nearest_m": round(
                    float(det.get("priority_xy_dist_from_nearest_m", 0.0)), 4
                ),
                "priority_center_dist_from_nearest_px": round(
                    float(det.get("priority_center_dist_from_nearest_px", 0.0)), 1
                ),
                "priority_xy_gate": bool(det.get("priority_xy_gate", False)),
                "priority_position_gate": bool(det.get("priority_position_gate", False)),
                "priority_depth_gate": bool(det.get("priority_depth_gate", False)),
                "priority_hole_conf_best": det.get("priority_hole_conf_best"),
                "priority_non_hole_conf_best": det.get("priority_non_hole_conf_best"),
                "priority_hole_conf_margin": round(
                    float(det.get("priority_hole_conf_margin", 0.0)), 3
                ),
                "priority_hole_conf_win": bool(det.get("priority_hole_conf_win", False)),
                "priority_hole_tie_applied": bool(det.get("priority_hole_tie_applied", False)),
                "priority_square_tie_applied": bool(det.get("priority_square_tie_applied", False)),
                "reason": (
                    f"camera-XY tie-break: compare candidates only when "
                    f"camera XY dist <= {self.priority_xy_tie_m:.3f}m, "
                    f"then depth diff <= {self.priority_depth_tie_m:.3f}m; "
                    f"inside that tie group, {self.priority_hole_class} is preferred only when "
                    f"its YOLO confidence >= best non-hole confidence + "
                    f"{self.priority_hole_conf_margin:.3f}"
                ),
            })

        return table

    def _log_priority_table(
        self,
        detections: List[Dict],
        trigger_seq: int,
        title: str = "PRIORITY_TABLE",
    ) -> None:
        """Print a readable candidate ranking table to ROS log."""
        if not detections:
            self.get_logger().info(f"[{title}] seq={trigger_seq} no detections")
            return

        lines = [
            f"[{title}] seq={trigger_seq} candidates={len(detections)} "
            f"rule=camera_xy_then_depth_hole_conf_compare_tie_break "
            f"xy_tie_m={self.priority_xy_tie_m:.3f} "
            f"depth_tie_m={self.priority_depth_tie_m:.3f} "
            f"hole_class={self.priority_hole_class} "
            f"hole_conf_margin={self.priority_hole_conf_margin:.3f}"
        ]

        for rank, det in enumerate(detections, start=1):
            cls = str(det.get("class_name", "unknown"))
            obj_id = int(self.class_to_id.get(cls, -1))
            depth_score = float(det.get("depth_score", det.get("depth_med", float("inf"))))
            depth_med = float(det.get("depth_med", float("inf")))
            conf = float(det.get("confidence", 0.0))

            tag = "SELECT" if rank == 1 else "      "
            lines.append(
                f"  #{rank:<2} {tag} "
                f"class={cls:<14} id={obj_id:<2} "
                f"depth_score={depth_score:.4f}m "
                f"depth_med={depth_med:.4f}m "
                f"conf={conf:.3f} "
                f"xy_dist={float(det.get('priority_xy_dist_from_nearest_m', 0.0)):.4f}m "
                f"px_dist={float(det.get('priority_center_dist_from_nearest_px', 0.0)):.1f}px "
                f"d_diff={float(det.get('priority_depth_diff_from_nearest_m', 0.0)):.4f}m "
                f"xy_gate={bool(det.get('priority_xy_gate', False))} "
                f"depth_gate={bool(det.get('priority_depth_gate', False))} "
                f"hole_best={det.get('priority_hole_conf_best')} "
                f"nonhole_best={det.get('priority_non_hole_conf_best')} "
                f"hole_win={bool(det.get('priority_hole_conf_win', False))} "
                f"hole_tie={bool(det.get('priority_hole_tie_applied', False))}"
            )

        self.get_logger().info("\n".join(lines))


    def _draw_depth_histogram_image(
        self,
        z: np.ndarray,
        title: str = "depth histogram",
        width: int = 900,
        height: int = 500,
        bins: int = 80,
    ) -> np.ndarray:
        """Create a simple depth histogram image using only OpenCV."""
        img = np.full((height, width, 3), 255, dtype=np.uint8)

        z = np.asarray(z, dtype=np.float32)
        z = z[np.isfinite(z)]
        z = z[z > 0]

        if len(z) == 0:
            cv2.putText(img, "No valid depth", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return img

        z_min = float(np.percentile(z, 1))
        z_max = float(np.percentile(z, 99))
        if z_max <= z_min:
            z_min = float(np.min(z))
            z_max = float(np.max(z)) + 1e-6

        hist, _ = np.histogram(z, bins=bins, range=(z_min, z_max))

        left, right = 70, width - 30
        top, bottom = 80, height - 70
        plot_w = right - left
        plot_h = bottom - top

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), 1)
        max_count = int(hist.max()) if hist.max() > 0 else 1

        for i, count in enumerate(hist):
            x0 = int(left + i * plot_w / bins)
            x1 = int(left + (i + 1) * plot_w / bins)
            bar_h = int((count / max_count) * plot_h)
            y0 = bottom - bar_h
            cv2.rectangle(img, (x0, y0), (x1, bottom), (80, 80, 80), -1)

        percentiles = {
            "p05": float(np.percentile(z, 5)),
            "p10": float(np.percentile(z, 10)),
            "p20": float(np.percentile(z, 20)),
            "p35": float(np.percentile(z, 35)),
            "p50": float(np.percentile(z, 50)),
        }

        for name, value in percentiles.items():
            x = int(left + (value - z_min) / max(z_max - z_min, 1e-6) * plot_w)
            x = int(np.clip(x, left, right))
            cv2.line(img, (x, top), (x, bottom), (0, 0, 255), 2)
            cv2.putText(img, f"{name}:{value:.3f}", (x + 3, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(img, title[:100], (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(
            img,
            f"range: {z_min:.3f}m ~ {z_max:.3f}m | n={len(z)}",
            (left, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        return img

    def _save_priority_debug_artifacts(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        display: np.ndarray,
        detections: List[Dict],
        trigger_seq: int,
        selected_det: Optional[Dict] = None,
    ) -> None:
        """Save per-trigger mask/depth artifacts and overwrite previous files.

        This function intentionally deletes priority_debug_dir every trigger so that
        debug data does not accumulate across runs.
        """
        if not self.priority_debug_save:
            return

        out_dir = self.priority_debug_dir

        try:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            masks_dir = out_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_dir / "color.png"), color)

            depth_m = depth.astype(np.float32) * float(self.depth_scale)
            valid = depth_m > 0
            if np.any(valid):
                vmin = float(np.percentile(depth_m[valid], 2))
                vmax = float(np.percentile(depth_m[valid], 98))
                depth_norm = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            else:
                depth_norm = np.zeros_like(depth_m, dtype=np.float32)

            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            depth_color[~valid] = (0, 0, 0)
            cv2.imwrite(str(out_dir / "depth_colormap.png"), depth_color)
            cv2.imwrite(str(out_dir / "overlay_priority.png"), display)

            table = []
            for rank, det in enumerate(detections, start=1):
                cls = str(det.get("class_name", "unknown"))
                conf = float(det.get("confidence", 0.0))
                obj_id = int(self.class_to_id.get(cls, -1))
                is_selected = bool(det is selected_det)
                safe_cls = cls.replace("/", "_").replace(" ", "_")
                prefix = f"rank{rank:02d}_{safe_cls}"

                mask_img = det["mask_img"]
                mask_bool = mask_img > 0

                cv2.imwrite(str(masks_dir / f"{prefix}_mask.png"), mask_img)

                mask_color = np.zeros_like(color)
                mask_color[mask_bool] = self.class_colors.get(cls, (255, 255, 255))
                mask_overlay = cv2.addWeighted(color, 0.65, mask_color, 0.35, 0)
                cv2.imwrite(str(masks_dir / f"{prefix}_overlay.png"), mask_overlay)

                det_depth_color = np.zeros_like(depth_color)
                det_depth_color[mask_bool] = depth_color[mask_bool]
                cv2.imwrite(str(masks_dir / f"{prefix}_depth_colormap.png"), det_depth_color)

                z = depth[mask_bool].astype(np.float32) * float(self.depth_scale)
                z = z[z > 0]
                hist_path = masks_dir / f"{prefix}_hist.png"
                if len(z) > 0:
                    hist_img = self._draw_depth_histogram_image(
                        z,
                        title=(
                            f"#{rank} {cls} "
                            f"score={det.get('depth_score', float('inf')):.4f}m "
                            f"med={det.get('depth_med', float('inf')):.4f}m"
                        ),
                    )
                    cv2.imwrite(str(hist_path), hist_img)

                ply_path = masks_dir / f"{prefix}_points.ply"
                ply_count = save_mask_point_cloud_ply(
                    ply_path,
                    depth,
                    mask_img,
                    color,
                    self.intrinsics,
                    self.depth_scale,
                )

                stats = depth_stats_in_mask(depth, mask_img, self.depth_scale)
                table.append({
                    "rank": int(rank),
                    "selected": is_selected,
                    "class_name": cls,
                    "class_id": obj_id,
                    "confidence": round(conf, 3),
                    "depth_score_m": round(float(det.get("depth_score", float("inf"))), 4),
                    "depth_median_m": round(float(det.get("depth_med", float("inf"))), 4),
                    "bbox": [int(v) for v in det.get("bbox", [0, 0, 0, 0])],
                    "depth_stats": stats,
                    "pointcloud_points_saved": int(ply_count),
                    "files": {
                        "mask": str(masks_dir / f"{prefix}_mask.png"),
                        "overlay": str(masks_dir / f"{prefix}_overlay.png"),
                        "depth_colormap": str(masks_dir / f"{prefix}_depth_colormap.png"),
                        "histogram": str(hist_path),
                        "pointcloud": str(ply_path),
                    },
                })

            debug_json = {
                "trigger_seq": int(trigger_seq),
                "priority_rule": (
                    f"camera-XY tie-break: compare camera XY distance first; "
                    f"if XY dist <= {self.priority_xy_tie_m:.3f}m, "
                    f"then compare depth diff <= {self.priority_depth_tie_m:.3f}m; "
                    f"inside that tie group, prefer {self.priority_hole_class} only when "
                    f"hole confidence >= best non-hole confidence + "
                    f"{self.priority_hole_conf_margin:.3f}"
                ),
                "priority_depth_tie_m": float(self.priority_depth_tie_m),
                "priority_xy_tie_m": float(self.priority_xy_tie_m),
                "priority_hole_class": str(self.priority_hole_class),
                "priority_hole_conf_margin": float(self.priority_hole_conf_margin),
                "depth_score_definition": "median of nearest 35 percent valid depth values inside each mask",
                "depth_scale": float(self.depth_scale),
                "debug_dir": str(out_dir),
                "overwrite_mode": True,
                "selected_rank": None,
                "selected_class": None,
                "detections": table,
            }

            if selected_det is not None:
                for row in table:
                    if row["selected"]:
                        debug_json["selected_rank"] = row["rank"]
                        debug_json["selected_class"] = row["class_name"]
                        break

            with open(out_dir / "priority_table.json", "w") as f:
                json.dump(debug_json, f, indent=2, ensure_ascii=False)

            self.get_logger().info(
                f"[PRIORITY_DEBUG] saved trigger_seq={trigger_seq} "
                f"candidates={len(detections)} dir={out_dir}"
            )
        except Exception as e:
            self.get_logger().warn(f"[PRIORITY_DEBUG] save failed: {e}")
            traceback.print_exc()


    def _find_detection_for_last_pose(self, detections: List[Dict]) -> Optional[Dict]:
        """Find current detection corresponding to the last successful object pose.

        현재는 object tracking ID가 없으므로, 마지막 성공 class와 같은 detection 중
        가장 가까운 것을 last pose overlay 대상으로 사용한다.
        """
        if self.last_object_class_name is None:
            return None
        candidates = [d for d in detections if d["class_name"] == self.last_object_class_name]
        if not candidates:
            return None
        candidates = self._sort_detections_with_square_tie_break(candidates)
        return candidates[0]

    def _get_empty_space_roi_mask(self, image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, Dict]:
        """Build fixed rectangular ROI mask for empty-space proposals."""
        h, w = image_shape[:2]

        x_min = int(self.get_parameter("empty_roi_x_min").value)
        y_min = int(self.get_parameter("empty_roi_y_min").value)
        x_max_param = int(self.get_parameter("empty_roi_x_max").value)
        y_max_param = int(self.get_parameter("empty_roi_y_max").value)
        right_margin = int(self.get_parameter("empty_roi_right_margin").value)
        bottom_margin = int(self.get_parameter("empty_roi_bottom_margin").value)

        x_max = x_max_param if x_max_param >= 0 else (w - right_margin)
        y_max = y_max_param if y_max_param >= 0 else (h - bottom_margin)

        x_min = int(np.clip(x_min, 0, w - 1))
        y_min = int(np.clip(y_min, 0, h - 1))
        x_max = int(np.clip(x_max, x_min + 1, w))
        y_max = int(np.clip(y_max, y_min + 1, h))

        roi = np.zeros((h, w), dtype=np.uint8)
        roi[y_min:y_max, x_min:x_max] = 255

        meta = {
            "type": "rect",
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }
        return roi, meta

    def _build_occupied_mask_from_detections(
        self,
        detections: List[Dict],
        image_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Union object masks and make a dilated near-object exclusion mask."""
        h, w = image_shape[:2]
        occupied = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            mask_img = det.get("mask_img")
            if mask_img is None:
                continue
            occupied[mask_img > 0] = 255

        dilate_px = max(0, int(self.get_parameter("empty_mask_dilate_px").value))
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            occupied_near = cv2.dilate(occupied, kernel, iterations=1)
        else:
            occupied_near = occupied.copy()

        return occupied, occupied_near

    def _make_object_translation_table(
        self,
        detections: List[Dict],
        depth: np.ndarray,
    ) -> List[Dict]:
        """Return detected object translations only: class/id/conf + camera xyz."""
        objects = []

        for rank, det in enumerate(detections, start=1):
            cls = str(det.get("class_name", "unknown"))
            obj_id = int(self.class_to_id.get(cls, -1))
            t_cam = estimate_mask_translation_camera(
                depth,
                det.get("mask_img"),
                self.intrinsics,
                self.depth_scale,
            )

            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            row = {
                "rank": int(rank),
                "class_name": cls,
                "class_id": obj_id,
                "confidence": round(float(det.get("confidence", 0.0)), 3),
                "bbox": [x1, y1, x2, y2],
                "depth_score_m": round(float(det.get("depth_score", float("inf"))), 4),
                "depth_median_m": round(float(det.get("depth_med", float("inf"))), 4),
                "camera_translation_valid": bool(t_cam is not None),
                "camera_translation": None,
            }

            if t_cam is not None:
                row["camera_translation"] = {
                    "x": round(float(t_cam[0]), 4),
                    "y": round(float(t_cam[1]), 4),
                    "z": round(float(t_cam[2]), 4),
                }

            objects.append(row)

        return objects

    def _generate_empty_space_candidates(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        detections: List[Dict],
        trigger_seq: int,
    ) -> Dict:
        """Generate image-grid empty-space candidates.

        Stage handled here:
          1) fixed image-grid proposals
          2) fixed ROI filtering
          3) object mask overlap / near-object filtering
          4) camera-coordinate candidate point from local depth patch
          5) detected object translations in camera frame

        World-frame object-size filtering is intentionally left to the calib node.
        """
        h, w = color.shape[:2]
        step_px = max(5, int(self.get_parameter("empty_grid_step_px").value))
        max_candidates = max(1, int(self.get_parameter("empty_max_candidates").value))
        patch_radius = max(1, int(self.get_parameter("empty_depth_patch_radius_px").value))
        valid_ratio_min = float(self.get_parameter("empty_depth_valid_ratio_min").value)
        spread_max_m = float(self.get_parameter("empty_depth_spread_max_m").value)

        roi_mask, roi_meta = self._get_empty_space_roi_mask(color.shape)
        occupied_mask, occupied_near_mask = self._build_occupied_mask_from_detections(detections, color.shape)

        # Distance to the nearest excluded object-mask region.
        free_for_distance = (occupied_near_mask == 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(free_for_distance, cv2.DIST_L2, 5)

        proposals_total = 0
        removed_outside_roi = 0
        removed_on_object_mask = 0
        removed_near_object_mask = 0
        removed_bad_depth = 0
        raw_candidates = []

        # Use half-step offset so grid points are centered inside cells.
        for v in range(step_px // 2, h, step_px):
            for u in range(step_px // 2, w, step_px):
                proposals_total += 1

                if roi_mask[v, u] == 0:
                    removed_outside_roi += 1
                    continue

                if occupied_mask[v, u] > 0:
                    removed_on_object_mask += 1
                    continue

                if occupied_near_mask[v, u] > 0:
                    removed_near_object_mask += 1
                    continue

                dstat = patch_depth_stats_at_pixel(
                    depth,
                    u,
                    v,
                    self.depth_scale,
                    radius_px=patch_radius,
                )

                if (
                    (not dstat["valid"])
                    or float(dstat["valid_ratio"]) < valid_ratio_min
                    or (
                        dstat["spread_p90_p10_m"] is not None
                        and float(dstat["spread_p90_p10_m"]) > spread_max_m
                    )
                ):
                    removed_bad_depth += 1
                    continue

                p_cam = deproject_pixel_to_camera(
                    u,
                    v,
                    float(dstat["median_m"]),
                    self.intrinsics,
                )

                if p_cam is None:
                    removed_bad_depth += 1
                    continue

                # Prefer larger clearance, with a very small center preference.
                clearance_px = float(dist_map[v, u])
                du = (float(u) - (w / 2.0)) / max(w / 2.0, 1.0)
                dv = (float(v) - (h / 2.0)) / max(h / 2.0, 1.0)
                center_penalty = float(np.sqrt(du * du + dv * dv))
                score = clearance_px - 3.0 * center_penalty

                raw_candidates.append({
                    "u": int(u),
                    "v": int(v),
                    "score": float(score),
                    "clearance_px": clearance_px,
                    "camera_point": p_cam,
                    "depth_stats": dstat,
                })

        raw_candidates.sort(key=lambda c: c["score"], reverse=True)

        candidates = []
        for rank, cand in enumerate(raw_candidates[:max_candidates], start=1):
            p_cam = cand["camera_point"]
            dstat = cand["depth_stats"]
            candidates.append({
                "rank": int(rank),
                "pixel": {
                    "u": int(cand["u"]),
                    "v": int(cand["v"]),
                },
                "camera": {
                    "x": round(float(p_cam[0]), 4),
                    "y": round(float(p_cam[1]), 4),
                    "z": round(float(p_cam[2]), 4),
                },
                "score": round(float(cand["score"]), 3),
                "clearance_px": round(float(cand["clearance_px"]), 2),
                "depth_source": "patch_median",
                "depth_median_m": round(float(dstat["median_m"]), 4),
                "depth_valid_ratio": round(float(dstat["valid_ratio"]), 3),
                "depth_spread_p90_p10_m": (
                    None
                    if dstat["spread_p90_p10_m"] is None
                    else round(float(dstat["spread_p90_p10_m"]), 4)
                ),
                "status": "vision_valid_world_filter_pending",
            })

        objects = self._make_object_translation_table(detections, depth)

        return {
            "mode": "empty_space",
            "frame_id": self.frame_id,
            "trigger_seq": int(trigger_seq),
            "proposal_rule": "fixed_image_grid_then_roi_then_mask_near_exclusion",
            "world_filter_policy": "calib_node_should_transform_candidates_and_objects_then_apply_object_size_filter",
            "roi": roi_meta,
            "params": {
                "grid_step_px": int(step_px),
                "max_candidates": int(max_candidates),
                "mask_dilate_px": int(self.get_parameter("empty_mask_dilate_px").value),
                "depth_patch_radius_px": int(patch_radius),
                "depth_valid_ratio_min": float(valid_ratio_min),
                "depth_spread_max_m": float(spread_max_m),
            },
            "stats": {
                "proposals_total": int(proposals_total),
                "removed_outside_roi": int(removed_outside_roi),
                "removed_on_object_mask": int(removed_on_object_mask),
                "removed_near_object_mask": int(removed_near_object_mask),
                "removed_bad_depth": int(removed_bad_depth),
                "candidates_before_limit": int(len(raw_candidates)),
                "candidates_published": int(len(candidates)),
                "objects_published": int(len(objects)),
            },
            "objects_camera": objects,
            "candidates": candidates,
            "target": candidates[0] if candidates else None,
        }

    def _publish_empty_space_candidates(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        display: np.ndarray,
        detections: List[Dict],
        trigger_seq: int,
    ) -> Optional[Dict]:
        if not bool(self.get_parameter("empty_space_enable").value):
            return None

        data = self._generate_empty_space_candidates(
            color=color,
            depth=depth,
            detections=detections,
            trigger_seq=trigger_seq,
        )

        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.empty_space_pub.publish(msg)

        # Save this result so the OpenCV visualization remains visible for a few seconds.
        self.last_empty_space_debug = data
        self.last_empty_space_debug_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

        # Visualization on the trigger frame: green = published candidate, red = detected object translation.
        for cand in data.get("candidates", []):
            u = int(cand["pixel"]["u"])
            v = int(cand["pixel"]["v"])
            rank = int(cand["rank"])
            cv2.circle(display, (u, v), 5, (0, 255, 0), -1)
            if rank <= 5:
                cv2.putText(
                    display,
                    f"E{rank}",
                    (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

        for obj in data.get("objects_camera", []):
            t = obj.get("camera_translation")
            if not t:
                continue
            z = float(t["z"])
            if z <= 0:
                continue
            u = int(float(t["x"]) * self.intrinsics.fx / z + self.intrinsics.ppx)
            v = int(float(t["y"]) * self.intrinsics.fy / z + self.intrinsics.ppy)
            cv2.circle(display, (u, v), 6, (0, 0, 255), -1)

        self.get_logger().info(
            f"[EMPTY_SPACE] seq={trigger_seq} "
            f"candidates={data['stats']['candidates_published']} "
            f"objects={data['stats']['objects_published']} "
            f"topic={str(self.get_parameter('empty_space_topic').value)} "
            f"bundled_into_object_json=True"
        )

        return data


    def _draw_last_empty_space_debug(self, display: np.ndarray) -> None:
        """Keep the last empty-space proposal visualization visible for N seconds.

        This is intentionally independent from the trigger frame drawing so that
        candidates do not disappear too quickly in the OpenCV preview window.
        """
        if self.last_empty_space_debug is None or self.last_empty_space_debug_stamp_sec is None:
            return

        hold_sec = float(self.get_parameter("empty_space_vis_hold_sec").value)
        if hold_sec <= 0.0:
            self.last_empty_space_debug = None
            self.last_empty_space_debug_stamp_sec = None
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = float(now_sec - float(self.last_empty_space_debug_stamp_sec))

        if elapsed > hold_sec:
            self.last_empty_space_debug = None
            self.last_empty_space_debug_stamp_sec = None
            return

        data = self.last_empty_space_debug
        h, w = display.shape[:2]

        # ROI rectangle: blue.
        roi = data.get("roi", {})
        if roi.get("type") == "rect":
            x_min = int(np.clip(int(roi.get("x_min", 0)), 0, w - 1))
            y_min = int(np.clip(int(roi.get("y_min", 0)), 0, h - 1))
            x_max = int(np.clip(int(roi.get("x_max", w - 1)), 0, w - 1))
            y_max = int(np.clip(int(roi.get("y_max", h - 1)), 0, h - 1))
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(
                display,
                "EMPTY ROI",
                (x_min + 5, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

        # Empty-space candidates: green. Top-10 get rank labels.
        candidates = data.get("candidates", [])
        for cand in candidates:
            pixel = cand.get("pixel", {})
            u = int(pixel.get("u", -1))
            v = int(pixel.get("v", -1))
            rank = int(cand.get("rank", 0))

            if u < 0 or v < 0 or u >= w or v >= h:
                continue

            radius = 7 if rank == 1 else 5
            thickness = -1 if rank <= 10 else 1
            cv2.circle(display, (u, v), radius, (0, 255, 0), thickness)

            if rank <= 10:
                cv2.putText(
                    display,
                    f"E{rank}",
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2,
                )

        # Detected object camera translations reprojected into the image: red.
        objects = data.get("objects_camera", [])
        for obj in objects:
            t = obj.get("camera_translation")
            if not t:
                continue

            z = float(t.get("z", 0.0))
            if z <= 0.0:
                continue

            u = int(float(t.get("x", 0.0)) * self.intrinsics.fx / z + self.intrinsics.ppx)
            v = int(float(t.get("y", 0.0)) * self.intrinsics.fy / z + self.intrinsics.ppy)
            if u < 0 or v < 0 or u >= w or v >= h:
                continue

            cv2.circle(display, (u, v), 7, (0, 0, 255), -1)
            cls = str(obj.get("class_name", "obj"))
            cv2.putText(
                display,
                cls,
                (u + 8, v + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
            )

        # Status text.
        stats = data.get("stats", {})
        cand_n = int(stats.get("candidates_published", len(candidates)))
        obj_n = int(stats.get("objects_published", len(objects)))
        cv2.putText(
            display,
            f"EMPTY SPACE DEBUG {elapsed:.1f}/{hold_sec:.1f}s | cand={cand_n} obj={obj_n}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


    def _process_object_mode(
        self,
        color,
        depth,
        display,
        run_fp: bool,
        requested_class: Optional[str],
        requested_allowed_ids: Optional[List[int]],
        trigger_seq: int,
    ) -> Tuple[List[Dict], Dict]:
        status = {
            "triggered": bool(run_fp),
            "trigger_seq": int(trigger_seq),
            "requested_class": requested_class,
            "requested_allowed_ids": requested_allowed_ids,
            "status": "preview" if not run_fp else "pending",
        }
        trigger_text = "TRIGGERED" if run_fp else "WAIT_TRIGGER"
        cv2.putText(
            display,
            f"MODE: object | FP: {'ON' if self.fp_available else 'OFF'} | {trigger_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        # 1) object YOLO segmentation은 매 프레임 전체 detection을 수집하고 전부 시각화한다.
        raw_detections = self._collect_detections(self.object_model, color, depth)
        status["detected_count_raw"] = len(raw_detections)

        # On a 6D trigger, generate empty-space candidates before FoundationPose.
        # The same data is also bundled into the /object_poses JSON so the transform node
        # receives pose + empty-space candidates atomically from the same trigger/frame.
        empty_space_data = None
        if run_fp:
            empty_space_data = self._publish_empty_space_candidates(
                color=color,
                depth=depth,
                display=display,
                detections=raw_detections,
                trigger_seq=trigger_seq,
            )
            if empty_space_data is not None:
                status["empty_space"] = empty_space_data
                status["empty_space_bundle_policy"] = "included_in_object_poses_json_same_trigger"

        # Show current top-priority candidate in the preview window.
        # Current priority rule: smaller depth_score means closer object.
        if raw_detections:
            top = raw_detections[0]
            top_depth = float(top.get("depth_score", top["depth_med"]))
            cv2.putText(
                display,
                f"PRIORITY: #1 {top['class_name']} | d_score={top_depth:.3f}m | XY+depth+conf tie={self.priority_hole_class}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        available_ids, available_classes = self._available_object_info(raw_detections)
        status["available_ids"] = available_ids
        status["available_classes"] = available_classes

        if not raw_detections:
            status["status"] = "no_detection" if run_fp else "preview_no_detection"
            status["message"] = "no object detected"
            return [], status

        # 2) trigger 대상 선택용 detection list.
        #    requested_allowed_ids가 있으면 0인 물체는 제외하고,
        #    1인 물체들 중 depth가 가장 가까운 detection을 선택한다.
        target_detections = raw_detections
        if requested_allowed_ids is not None:
            allowed_ids_set = {int(v) for v in requested_allowed_ids}
            target_detections = [
                d for d in raw_detections
                if d["class_name"] in self.class_to_id
                and int(self.class_to_id[d["class_name"]]) in allowed_ids_set
            ]
        elif requested_class:
            target_detections = [
                d for d in raw_detections
                if d["class_name"] == requested_class
            ]

        # raw_detections is already sorted by priority, but sort again after filtering.
        target_detections = self._sort_detections_with_square_tie_break(target_detections)
        status["detected_count"] = len(target_detections)

        # Priority information for JSON/log debugging.
        priority_table = self._make_priority_table(target_detections)
        status["priority_rule"] = (
            f"smaller depth_score is closer and higher priority; "
            f"if camera XY dist <= {self.priority_xy_tie_m:.3f}m and "
            f"depth diff <= {self.priority_depth_tie_m:.3f}m, "
            f"{self.priority_hole_class} is preferred only when its confidence >= "
            f"best non-hole confidence + {self.priority_hole_conf_margin:.3f}"
        )
        status["priority_table"] = priority_table

        if run_fp:
            self._log_priority_table(target_detections, trigger_seq)

        # 3) 우선 전체 YOLO detection을 항상 그림.
        #    last pose가 붙을 detection은 중복으로 그리지 않고 뒤에서 pose overlay와 함께 강조한다.
        last_pose_det = self._find_detection_for_last_pose(raw_detections)
        for rank, det_vis in enumerate(raw_detections, start=1):
            if (last_pose_det is not None) and (det_vis is last_pose_det) and (not run_fp):
                continue

            depth_score = float(det_vis.get("depth_score", det_vis["depth_med"]))
            depth_med = float(det_vis["depth_med"])
            source_text = (
                f"#{rank} "
                f"{'SELECT' if rank == 1 else 'candidate'} | "
                f"d_score:{depth_score:.3f}m med:{depth_med:.3f}m"
            )
            self._draw_detection(display, det_vis, None, source_text)

        # 4) trigger가 없으면 마지막 성공 pose만 현재 detection 위에 유지 표시한다.
        if not run_fp:
            if last_pose_det is not None and self.last_object_pose_mat is not None:
                self._draw_detection(
                    display,
                    last_pose_det,
                    self.last_object_pose_mat,
                    self.last_object_status_text or "last FP pose",
                )

            status["status"] = "preview_detected"
            status["preview_class"] = raw_detections[0]["class_name"]
            status["preview_depth_m"] = round(float(raw_detections[0].get("depth_score", raw_detections[0]["depth_med"])), 4)
            status["preview_depth_median_m"] = round(float(raw_detections[0]["depth_med"]), 4)
            if self.last_object_class_name is not None:
                status["last_pose_class"] = self.last_object_class_name
            return [], status

        # 5) trigger가 들어왔는데 요청 class가 현재 안 보이면, 전체 preview는 유지하고 실패 반환.
        if not target_detections:
            status["status"] = "no_detection"
            if requested_allowed_ids is not None:
                status["message"] = f"no allowed object detected: allowed_ids={requested_allowed_ids}"
            else:
                status["message"] = f"requested class not detected: {requested_class}"
            return [], status

        # 6) FoundationPose는 선택된 1개 target만 수행한다.
        det = target_detections[0]
        class_name = det["class_name"]

        # Save per-trigger mask/depth debug artifacts.
        # Files are overwritten every trigger in ./debug_priority by default.
        self._save_priority_debug_artifacts(
            color=color,
            depth=depth,
            display=display,
            detections=target_detections,
            trigger_seq=trigger_seq,
            selected_det=det,
        )

        pose_mat = None
        refined_frames = 0
        register_iter = int(self.get_parameter("fp_register_iter").value)
        track_iter = int(self.get_parameter("fp_track_iter").value)
        track_loss_thr = float(self.get_parameter("fp_track_loss_thr").value)
        trigger_track_frames = max(0, int(self.get_parameter("fp_trigger_track_frames").value))
        trigger_track_use_new_frames = bool(self.get_parameter("fp_trigger_track_use_new_frames").value)

        try:
            est = self._get_fp_estimator(class_name)
            if est is not None:
                # Each trigger starts with a fresh register to avoid stale pose.
                # Then, only inside this trigger request, run a short track_one refinement.
                # This is useful when robot/camera/object are static during capture.
                est.reset()
                pose_mat = est.estimate(
                    rgb_bgr=color,
                    depth_uint16=depth,
                    mask_uint8=det["mask_img"],
                    K=self.K,
                    register_iter=register_iter,
                    track_iter=track_iter,
                    track_loss_thr=track_loss_thr,
                    use_tracking=False,
                    depth_scale=self.depth_scale,
                )

                for _ in range(trigger_track_frames):
                    track_color = color
                    track_depth = depth
                    if trigger_track_use_new_frames:
                        next_color, next_depth = self._read_aligned_rgbd()
                        if next_color is not None and next_depth is not None:
                            track_color, track_depth = next_color, next_depth

                    pose_mat = est.estimate(
                        rgb_bgr=track_color,
                        depth_uint16=track_depth,
                        mask_uint8=det["mask_img"],
                        K=self.K,
                        register_iter=register_iter,
                        track_iter=track_iter,
                        track_loss_thr=track_loss_thr,
                        use_tracking=True,
                        depth_scale=self.depth_scale,
                    )
                    refined_frames += 1

                est.reset()
        except Exception as e:
            self.get_logger().warn(f"[FP] triggered pose failed: {class_name}: {e}")
            traceback.print_exc()
            if class_name in self._fp_estimators:
                self._fp_estimators[class_name].reset()
        if pose_mat is None:
            self._draw_detection(display, det, None, "FP failed")
            status["status"] = "fp_failed"
            status["message"] = f"FoundationPose failed for {class_name}"
            status["selected_class"] = class_name
            return [], status

        extra = {
            "pose_source": "foundationpose_register_short_track",
            "trigger_seq": int(trigger_seq),
            "register_iter": int(register_iter),
            "track_iter": int(track_iter),
            "track_refine_frames": int(refined_frames),
            "priority": {
                "selected": True,
                "reason": (
                    "allowed_ids_nearest_depth_with_camera_xy_depth_hole_conf_compare_tie_break"
                    if requested_allowed_ids is not None
                    else (
                        "nearest_depth_with_camera_xy_depth_hole_conf_compare_tie_break"
                        if requested_class is None
                        else "requested_class_nearest_depth_with_camera_xy_depth_hole_conf_compare_tie_break"
                    )
                ),
                "priority_depth_tie_m": round(float(self.priority_depth_tie_m), 4),
                "priority_xy_tie_m": round(float(self.priority_xy_tie_m), 4),
                "priority_hole_class": str(self.priority_hole_class),
                "priority_hole_conf_margin": round(float(self.priority_hole_conf_margin), 4),
                "allowed_ids": requested_allowed_ids,
                "depth_median_m": round(float(det["depth_med"]), 4),
                "depth_score_m": round(float(det.get("depth_score", det["depth_med"])), 4),
                "detected_count": len(target_detections),
                "detected_count_raw": int(status["detected_count_raw"]),
            },
        }
        obj = pose_to_dict(pose_mat, class_name, det["confidence"], extra)

        ps = make_pose_stamped(pose_mat, self.frame_id)
        ps.header.stamp = self.get_clock().now().to_msg()
        self.object_pose_pub.publish(ps)

        # 7) 새 trigger 결과로 last pose를 즉시 교체한다.
        self.last_object_pose_mat = pose_mat.copy()
        self.last_object_class_name = class_name
        self.last_object_status_text = f"last FP pose seq={trigger_seq}"

        # 8) trigger 대상은 pose axis + 3D CAD bbox로 다시 강조해서 그림.
        self._draw_detection(display, det, pose_mat, f"TRIGGERED reg+track({refined_frames})")

        selected_rank = target_detections.index(det) + 1

        status["status"] = "success"
        status["selected_class"] = class_name
        status["selected_id"] = int(self.class_to_id[class_name])
        status["selected_rank"] = int(selected_rank)
        status["selected_depth_m"] = round(float(det.get("depth_score", det["depth_med"])), 4)
        status["selected_depth_median_m"] = round(float(det["depth_med"]), 4)
        self.get_logger().info(
            f"[OBJECT_6D_SELECTED] seq={trigger_seq} "
            f"rank=#{selected_rank} "
            f"class={class_name} id={self.class_to_id[class_name]} "
            f"reason=nearest_depth_score_with_camera_xy_depth_hole_conf_compare_tie_break "
            f"depth_score={det.get('depth_score', det['depth_med']):.4f}m "
            f"depth_med={det['depth_med']:.4f}m "
            f"conf={det['confidence']:.3f} "
            f"register_iter={register_iter} "
            f"track_iter={track_iter} "
            f"refined_frames={refined_frames}"
        )
        return [obj], status

    def _estimate_yaw_from_template(self, mask_img: np.ndarray, class_name: str):
        if class_name in self.no_yaw_classes:
            return 0.0, 1.0, "circle_no_yaw"
        if class_name not in self.rotated_templates:
            return 0.0, 0.0, "template_missing"
        crop = crop_mask_to_square(mask_img)
        if crop is None:
            return 0.0, 0.0, "mask_invalid"
        query = normalize_binary(crop, self.match_size)
        best_angle, best_score = 0.0, -1.0
        for angle, tmpl in self.rotated_templates[class_name]:
            score = iou_score(query, tmpl)
            if score > best_score:
                best_angle, best_score = angle, score
        return float(best_angle), float(best_score), "template_iou"

    def _process_insert_mode(self, color, depth, display) -> List[Dict]:
        cv2.putText(display, "MODE: insert | PCA+template", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        detections = self._collect_detections(self.insert_model, color, depth)
        objects = []
        for det in detections:
            points = get_point_cloud(depth, det["mask_img"], self.intrinsics, self.depth_scale)
            if points is None:
                continue
            pose_mat = estimate_pca_pose(points)
            yaw_deg, yaw_score, yaw_source = self._estimate_yaw_from_template(det["mask_img"], det["class_name"])
            obj = pca_pose_to_dict(pose_mat, det["class_name"], det["confidence"], yaw_deg, yaw_score, yaw_source)
            obj["pose_source"] = "depth_pca_template"
            obj["depth_median_m"] = round(float(det["depth_med"]), 4)
            obj["depth_score_m"] = round(float(det.get("depth_score", det["depth_med"])), 4)
            objects.append(obj)
            ps = make_pose_stamped(pose_mat, self.frame_id)
            ps.header.stamp = self.get_clock().now().to_msg()
            self.insert_pose_pub.publish(ps)
            self._draw_insert_detection(display, det, pose_mat, yaw_deg, yaw_score)
        return objects

    def _draw_detection(self, display, det, pose_mat, source_text):
        cls = det["class_name"]
        color_val = self.class_colors.get(cls, (255, 255, 255))
        overlay = display.copy()
        if det["mask_xy"] is not None and len(det["mask_xy"]) > 0:
            cv2.fillPoly(overlay, [det["mask_xy"].astype(np.int32)], color_val)
        display[:] = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), color_val, 3)
        label = f"TARGET {cls} {det['confidence']:.2f} | {source_text}"
        if pose_mat is not None:
            vis_info = self._mesh_vis_info.get(cls)
            if vis_info is not None:
                center_pose = np.asarray(pose_mat, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(vis_info["to_origin"])
                draw_projected_3d_bbox(
                    display,
                    center_pose,
                    self.K,
                    vis_info["bbox"],
                    color=(255, 255, 255),
                    thickness=2,
                )

            draw_pose_axis(display, pose_mat, self.K, axis_len=0.10)
            t = pose_mat[:3, 3]
            label += f" | X:{t[0]:+.3f} Y:{t[1]:+.3f} Z:{t[2]:.3f}m"
        cv2.putText(display, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_val, 2)

    def _draw_insert_detection(self, display, det, pose_mat, yaw_deg, yaw_score):
        cls = det["class_name"]
        color_val = self.class_colors.get(cls, (255, 255, 255))
        overlay = display.copy()
        if det["mask_xy"] is not None and len(det["mask_xy"]) > 0:
            cv2.fillPoly(overlay, [det["mask_xy"].astype(np.int32)], color_val)
        display[:] = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), color_val, 2)
        t = pose_mat[:3, 3]
        if t[2] > 0:
            cx = int(t[0] * self.intrinsics.fx / t[2] + self.intrinsics.ppx)
            cy = int(t[1] * self.intrinsics.fy / t[2] + self.intrinsics.ppy)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            if cls not in self.no_yaw_classes:
                length = 45
                yaw_rad = np.radians(yaw_deg)
                ex = int(cx + length * np.cos(yaw_rad))
                ey = int(cy - length * np.sin(yaw_rad))
                cv2.arrowedLine(display, (cx, cy), (ex, ey), (0, 0, 255), 2, tipLength=0.25)
        label = f"{cls} {det['confidence']:.2f} | X:{t[0]:+.2f} Y:{t[1]:+.2f} Z:{t[2]:.2f}m | yaw:{yaw_deg:.1f} score:{yaw_score:.2f}"
        cv2.putText(display, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_val, 2)

    def _publish_json(self, pub, mode: str, objects: List[Dict], extra: Optional[Dict] = None):
        msg = String()
        data = {
            "mode": mode,
            "target": objects[0] if objects else None,
            "objects": objects,
            "detected_count": len(objects),
        }
        if extra:
            data.update(extra)
        msg.data = json.dumps(data, ensure_ascii=False)
        pub.publish(msg)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.enable_visualization:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MixedPoseVisionNode()
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