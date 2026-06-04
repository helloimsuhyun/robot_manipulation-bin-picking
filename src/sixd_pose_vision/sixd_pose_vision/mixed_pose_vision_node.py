#!/usr/bin/env python3
"""
Mixed 6D pose vision node.

object mode: RealSense RGB-D -> YOLO-seg preview -> trigger topic -> nearest object -> FoundationPose register + short tracking refinement 6D pose
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
        self.declare_parameter("object_trigger_topic", "/object_6d_trigger")

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

        # Publishers/subscribers
        self.object_pub = self.create_publisher(String, str(self.get_parameter("object_topic").value), 10)
        self.insert_pub = self.create_publisher(String, str(self.get_parameter("insert_topic").value), 10)
        self.object_pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("object_pose_topic").value), 10)
        self.insert_pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("insert_pose_topic").value), 10)
        self.create_subscription(String, str(self.get_parameter("detect_mode_topic").value), self._mode_callback, 10)
        self.create_subscription(String, str(self.get_parameter("object_trigger_topic").value), self._object_trigger_callback, 10)

        # Object 6D is computed only when a trigger message arrives.
        # Trigger payload examples:
        #   "" or "nearest"          -> nearest detected object
        #   "cross"                  -> nearest detected cross only
        #   '{"class":"cylinder"}' -> nearest detected cylinder only
        self._object_trigger_pending = False
        self._object_trigger_class: Optional[str] = None
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

        if payload:
            # Accept either raw class name, "nearest", or JSON: {"class":"cross"}
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    target_class = data.get("class") or data.get("target_class")
            except Exception:
                if payload.lower() not in {"nearest", "object", "trigger", "capture", "1", "true"}:
                    target_class = payload

        if target_class is not None:
            target_class = str(target_class).strip()
            if not target_class:
                target_class = None

        self._object_trigger_pending = True
        self._object_trigger_class = target_class
        self._object_trigger_seq += 1
        self.get_logger().info(
            f"[OBJECT_TRIGGER] seq={self._object_trigger_seq} target_class={target_class or 'nearest'}"
        )

    def _consume_object_trigger(self) -> Tuple[bool, Optional[str], int]:
        if not self._object_trigger_pending:
            return False, None, self._object_trigger_seq
        target_class = self._object_trigger_class
        seq = self._object_trigger_seq
        self._object_trigger_pending = False
        self._object_trigger_class = None
        return True, target_class, seq

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
            triggered, target_class, trigger_seq = self._consume_object_trigger()
            objects, status = self._process_object_mode(
                color=color,
                depth=depth,
                display=display,
                run_fp=triggered,
                requested_class=target_class,
                trigger_seq=trigger_seq,
            )
            # Object 6D response is published only for an explicit trigger.
            if triggered:
                self._publish_json(self.object_pub, "object", objects, extra=status)
        else:
            objects = self._process_insert_mode(color, depth, display)
            self._publish_json(self.insert_pub, "insert", objects)

        if self.enable_visualization:
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
            detections.append({
                "class_name": results.names[cls_id],
                "confidence": float(results.boxes.conf[i]),
                "mask_xy": mask_xy,
                "mask_img": mask_img,
                "depth_med": depth_median_in_mask(depth, mask_img, self.depth_scale),
                "bbox": results.boxes.xyxy[i].cpu().numpy().astype(int),
            })
        detections.sort(key=lambda d: d["depth_med"])
        return detections

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
        candidates.sort(key=lambda d: d["depth_med"])
        return candidates[0]

    def _process_object_mode(
        self,
        color,
        depth,
        display,
        run_fp: bool,
        requested_class: Optional[str],
        trigger_seq: int,
    ) -> Tuple[List[Dict], Dict]:
        status = {
            "triggered": bool(run_fp),
            "trigger_seq": int(trigger_seq),
            "requested_class": requested_class,
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

        available_ids, available_classes = self._available_object_info(raw_detections)
        status["available_ids"] = available_ids
        status["available_classes"] = available_classes

        if not raw_detections:
            status["status"] = "no_detection" if run_fp else "preview_no_detection"
            status["message"] = "no object detected"
            return [], status

        # 2) trigger 대상 선택용 detection list. requested_class가 있으면 그 class 중 nearest.
        target_detections = raw_detections
        if requested_class:
            target_detections = [d for d in raw_detections if d["class_name"] == requested_class]
        status["detected_count"] = len(target_detections)

        # 3) 우선 전체 YOLO detection을 항상 그림.
        #    last pose가 붙을 detection은 중복으로 그리지 않고 뒤에서 pose overlay와 함께 강조한다.
        last_pose_det = self._find_detection_for_last_pose(raw_detections)
        for det_vis in raw_detections:
            if (last_pose_det is not None) and (det_vis is last_pose_det) and (not run_fp):
                continue
            self._draw_detection(display, det_vis, None, "detected")

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
            status["preview_depth_m"] = round(float(raw_detections[0]["depth_med"]), 4)
            if self.last_object_class_name is not None:
                status["last_pose_class"] = self.last_object_class_name
            return [], status

        # 5) trigger가 들어왔는데 요청 class가 현재 안 보이면, 전체 preview는 유지하고 실패 반환.
        if not target_detections:
            status["status"] = "no_detection"
            status["message"] = f"requested class not detected: {requested_class}"
            return [], status

        # 6) FoundationPose는 선택된 1개 target만 수행한다.
        det = target_detections[0]
        class_name = det["class_name"]

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
                "reason": "nearest_depth" if requested_class is None else "requested_class_nearest_depth",
                "depth_median_m": round(float(det["depth_med"]), 4),
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

        status["status"] = "success"
        status["selected_class"] = class_name
        status["selected_depth_m"] = round(float(det["depth_med"]), 4)
        self.get_logger().info(
            f"[OBJECT_6D] seq={trigger_seq} success class={class_name} "
            f"depth={det['depth_med']:.4f}m register_iter={register_iter} "
            f"track_iter={track_iter} refined_frames={refined_frames}"
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