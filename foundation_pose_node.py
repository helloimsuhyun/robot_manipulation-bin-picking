"""
FoundationPose 통합 6D Pose 추출 노드
======================================
파이프라인:
  RealSense RGBD
    → YOLO-seg (객체 감지 + mask)
    → Depth 기반 우선순위 정렬 (가까운 객체 먼저)
    → FoundationPose (mask + RGBD + CAD mesh → 정확한 6D pose)
    → ROS2 publish (/object_poses, /insert_poses)

실행:
  python foundation_pose_node.py

필수 준비:
  - ~/FoundationPose 설치 완료 (setup_foundationpose.sh 실행)
  - CAD/ 폴더에 cross.stl, cylinder.stl, hole.stl (미터 단위 권장)
"""

import sys
import os
import json
import numpy as np
import cv2
import trimesh
import pyrealsense2 as rs
from ultralytics import YOLO
from pathlib import Path
from scipy.spatial.transform import Rotation

# FoundationPose 경로 추가
FP_PATH = Path.home() / "FoundationPose"
sys.path.insert(0, str(FP_PATH))

try:
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    FOUNDATIONPOSE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] FoundationPose 로드 실패: {e}")
    print("       setup_foundationpose.sh 를 먼저 실행하세요.")
    FOUNDATIONPOSE_AVAILABLE = False

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MESH_DIRS = [BASE_DIR / "CAD", BASE_DIR / "meshes"]
MESH_EXTS = [".stl", ".obj", ".ply"]
MESH_SCALE = float(os.environ.get("CAD_MESH_SCALE", "0.001"))

MODELS = {
    "object": {
        "yolo_path": str(BASE_DIR / "runs/segment/train/weights/best.pt"),
        "classes": ["cross", "cylinder", "hole"],
        "colors": {
            "cross":    (0, 220, 0),
            "cylinder": (0, 140, 255),
            "hole":     (220, 0, 220),
        },
        "topic_json":  "/object_poses",
        "topic_pose":  "/object_pose_stamped",
    },
    "insert": {
        "yolo_path": str(BASE_DIR / "runs/segment/insert_seg/weights/best.pt"),
        "classes": ["cross_insert", "cylinder_insert", "hole_insert"],
        "colors": {
            "cross_insert":    (0, 220, 0),
            "cylinder_insert": (0, 140, 255),
            "hole_insert":     (220, 0, 220),
        },
        "topic_json":  "/insert_poses",
        "topic_pose":  "/insert_pose_stamped",
    },
}

# insert 클래스의 CAD 메쉬는 object 클래스 메쉬 재사용
MESH_ALIAS = {
    "cross_insert":    "cross",
    "cylinder_insert": "cylinder",
    "hole_insert":     "hole",
}

CONF_THRESH = 0.4
FP_REGISTER_ITER = 5   # 첫 등록 시 반복 (정확도 ↑, 속도 ↓)
FP_TRACK_ITER    = 2   # 트래킹 반복 (속도 중시)
TRACK_LOSS_THR   = 0.2 # score 이 값 이하면 재등록


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def load_mesh(class_name: str):
    """CAD 메쉬 로드. CAD_MESH_SCALE로 단위 보정 가능."""
    name = MESH_ALIAS.get(class_name, class_name)
    candidates = [mesh_dir / f"{name}{ext}" for mesh_dir in MESH_DIRS for ext in MESH_EXTS]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"메쉬 없음: {name}\n  검색 경로:\n  {searched}")

    mesh = trimesh.load(str(path), force="mesh")
    if MESH_SCALE != 1.0:
        mesh.apply_scale(MESH_SCALE)
    return mesh


def build_K(intrinsics) -> np.ndarray:
    """RealSense intrinsics → 3×3 카메라 행렬"""
    return np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1             ],
    ], dtype=np.float64)


def mask_from_polygon(mask_xy, shape):
    """YOLO polygon → binary mask (H×W uint8)"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [mask_xy.astype(np.int32)], 255)
    return mask


def depth_median_in_mask(depth_image, mask):
    """mask 영역 depth 중앙값 (m). depth=0 픽셀 제외."""
    z = depth_image[mask > 0].astype(float) * 0.001
    z = z[z > 0]
    return float(np.median(z)) if len(z) > 0 else float("inf")


def pose_to_dict(pose_mat: np.ndarray, class_name: str, confidence: float) -> dict:
    """4×4 pose matrix → JSON-직렬화 가능 dict"""
    t = pose_mat[:3, 3]
    R = pose_mat[:3, :3]
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return {
        "class":      class_name,
        "confidence": round(confidence, 3),
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


def make_pose_stamped(pose_mat: np.ndarray, frame_id: str = "camera_color_optical_frame") -> PoseStamped:
    """4×4 pose matrix → geometry_msgs/PoseStamped"""
    t = pose_mat[:3, 3]
    R = pose_mat[:3, :3]
    quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x    = float(t[0])
    msg.pose.position.y    = float(t[1])
    msg.pose.position.z    = float(t[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def draw_pose_axis(image, pose_mat, K, axis_len=0.05):
    """카메라 이미지에 3D 좌표축 투영 시각화"""
    t  = pose_mat[:3, 3]
    R  = pose_mat[:3, :3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def proj(pt3d):
        x, y, z = pt3d
        if z <= 0:
            return None
        return (int(x * fx / z + cx), int(y * fy / z + cy))

    o = proj(t)
    if o is None:
        return
    axes = [(R[:, 0], (0, 0, 255)), (R[:, 1], (0, 255, 0)), (R[:, 2], (255, 0, 0))]
    for axis_vec, color in axes:
        p = proj(t + axis_vec * axis_len)
        if p:
            cv2.arrowedLine(image, o, p, color, 2, tipLength=0.2)


# ---------------------------------------------------------------------------
# FoundationPose 래퍼
# ---------------------------------------------------------------------------

class FPEstimator:
    """
    단일 객체 클래스용 FoundationPose 래퍼.
    처음 호출 시 register(), 이후 track_one() 자동 전환.
    VRAM 절약을 위해 GPU context는 공유.
    """

    def __init__(self, class_name: str, mesh: trimesh.Trimesh, glctx, scorer, refiner):
        self.class_name = class_name
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )
        self.registered = False
        self.last_score  = 1.0

    def estimate(self, rgb, depth, mask, K):
        """
        rgb   : (H,W,3) uint8 BGR
        depth : (H,W)   uint16 mm
        mask  : (H,W)   uint8 0/255
        K     : (3,3)   float64

        반환: 4×4 float64 pose matrix (카메라 기준)
        """
        rgb_fp    = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.uint8)
        depth_fp  = depth.astype(np.float32) * 0.001   # mm → m

        if not self.registered or self.last_score < TRACK_LOSS_THR:
            pose = self.estimator.register(
                K=K, rgb=rgb_fp, depth=depth_fp,
                ob_mask=mask, iteration=FP_REGISTER_ITER,
            )
            self.registered = True
        else:
            pose = self.estimator.track_one(
                rgb=rgb_fp, depth=depth_fp,
                K=K, iteration=FP_TRACK_ITER,
            )

        # score 저장 (트래킹 손실 감지용)
        if hasattr(self.estimator, "last_score"):
            self.last_score = float(self.estimator.last_score)

        return np.array(pose, dtype=np.float64)

    def reset(self):
        self.registered = False
        self.last_score  = 1.0


# ---------------------------------------------------------------------------
# ROS2 노드
# ---------------------------------------------------------------------------

class FoundationPoseNode(Node):

    def __init__(self):
        super().__init__("foundation_pose_node")

        # --- YOLO 모델 로드 ---
        self.yolo_models = {}
        for mode, cfg in MODELS.items():
            p = Path(cfg["yolo_path"])
            if p.exists():
                self.yolo_models[mode] = YOLO(str(p))
                self.get_logger().info(f"[YOLO] {mode} 로드: {p}")
            else:
                self.get_logger().warn(f"[YOLO] 모델 없음: {p}")

        self.mode = "object"

        # --- FoundationPose 공유 자원 ---
        self.fp_available = FOUNDATIONPOSE_AVAILABLE
        if self.fp_available:
            try:
                self.glctx   = dr.RasterizeCudaContext()
                self.scorer  = ScorePredictor()
                self.refiner = PoseRefinePredictor()
                self.get_logger().info("[FP] GPU context 초기화 완료")
            except Exception as e:
                self.get_logger().error(f"[FP] GPU 초기화 실패: {e}")
                self.fp_available = False

        # 클래스별 FPEstimator 캐시 (lazy init)
        self._fp_estimators: dict[str, FPEstimator] = {}

        # --- 퍼블리셔 ---
        self.pub_json = {
            mode: self.create_publisher(String, cfg["topic_json"], 10)
            for mode, cfg in MODELS.items()
        }
        self.pub_pose = {
            mode: self.create_publisher(PoseStamped, cfg["topic_pose"], 10)
            for mode, cfg in MODELS.items()
        }

        # 모드 전환 구독
        self.create_subscription(String, "/detect_mode", self._mode_cb, 10)

        # --- RealSense 설정 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,  30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        ).get_intrinsics()
        self.K     = build_K(intr)
        self.align = rs.align(rs.stream.color)

        self.timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz
        self.get_logger().info("FoundationPoseNode ready. 모드: object")

    # ------------------------------------------------------------------ #

    def _mode_cb(self, msg):
        mode = msg.data.strip().lower()
        if mode not in MODELS:
            self.get_logger().warn(f"알 수 없는 모드: {mode}")
            return
        if mode not in self.yolo_models:
            self.get_logger().warn(f"YOLO 모델 없음: {mode}")
            return
        self.mode = mode
        # 모드 전환 시 트래킹 리셋
        for est in self._fp_estimators.values():
            est.reset()
        self.get_logger().info(f"모드 전환 → {mode}")

    # ------------------------------------------------------------------ #

    def _get_fp_estimator(self, class_name: str) -> FPEstimator | None:
        """클래스별 FPEstimator lazy 생성."""
        if class_name in self._fp_estimators:
            return self._fp_estimators[class_name]
        try:
            mesh = load_mesh(class_name)
            est  = FPEstimator(class_name, mesh, self.glctx, self.scorer, self.refiner)
            self._fp_estimators[class_name] = est
            self.get_logger().info(f"[FP] estimator 생성: {class_name}")
            return est
        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            return None

    # ------------------------------------------------------------------ #

    def _timer_cb(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f:
            return

        color = np.asanyarray(color_f.get_data())   # BGR uint8
        depth = np.asanyarray(depth_f.get_data())   # uint16 mm
        display = color.copy()

        cfg        = MODELS[self.mode]
        yolo_model = self.yolo_models.get(self.mode)
        if yolo_model is None:
            return

        results = yolo_model(color, conf=CONF_THRESH, verbose=False)[0]

        # ---- 감지 결과 수집 + depth 우선순위 정렬 ----
        detections = []
        if results.masks is not None:
            for i, mask_xy in enumerate(results.masks.xy):
                mask_img   = mask_from_polygon(mask_xy, color.shape)
                depth_med  = depth_median_in_mask(depth, mask_img)
                cls_id     = int(results.boxes.cls[i])
                class_name = results.names[cls_id]
                confidence = float(results.boxes.conf[i])
                bbox       = results.boxes.xyxy[i].cpu().numpy().astype(int)

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "mask_xy":    mask_xy,
                    "mask_img":   mask_img,
                    "depth_med":  depth_med,
                    "bbox":       bbox,
                })

        # 가까운 순 정렬 후 pick 대상 1개만 선택
        detections.sort(key=lambda d: d["depth_med"])
        target_detection = detections[0] if detections else None

        # ---- FoundationPose 6D pose 추출 (pick 대상 1개만) ----
        objects = []
        cv2.putText(display, f"MODE: {self.mode}  FP: {'ON' if self.fp_available else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if target_detection is not None:
            det = target_detection
            class_name = det["class_name"]
            confidence = det["confidence"]
            mask_img   = det["mask_img"]
            mask_xy    = det["mask_xy"]
            bbox       = det["bbox"]
            color_val  = cfg["colors"].get(class_name, (255, 255, 255))

            pose_mat = None

            if self.fp_available:
                est = self._get_fp_estimator(class_name)
                if est is not None:
                    try:
                        pose_mat = est.estimate(color, depth, mask_img, self.K)
                    except Exception as e:
                        self.get_logger().warn(f"[FP] {class_name} pose 실패: {e}")
                        est.reset()

            # FoundationPose 실패 시 depth 기반 fallback
            if pose_mat is None:
                pose_mat = self._depth_fallback_pose(depth, mask_img)

            if pose_mat is not None:
                obj_dict = pose_to_dict(pose_mat, class_name, confidence)
                obj_dict["priority"] = {
                    "selected": True,
                    "reason": "nearest_depth",
                    "depth_median_m": round(float(det["depth_med"]), 4),
                    "detected_count": len(detections),
                }
                objects.append(obj_dict)

                # PoseStamped 퍼블리시
                ps = make_pose_stamped(pose_mat)
                ps.header.stamp = self.get_clock().now().to_msg()
                self.pub_pose[self.mode].publish(ps)

                # 시각화
                overlay = display.copy()
                if len(mask_xy) > 0:
                    cv2.fillPoly(overlay, [mask_xy.astype(np.int32)], color_val)
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), color_val, 3)
                draw_pose_axis(display, pose_mat, self.K)

                t = pose_mat[:3, 3]
                label = (f"TARGET {class_name} {confidence:.2f} | "
                         f"X:{t[0]:+.3f} Y:{t[1]:+.3f} Z:{t[2]:.3f}m")
                cv2.putText(display, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_val, 2)
                cv2.putText(display, "#1 nearest", (x1 + 2, y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("FoundationPose 6D (ESC to quit)", display)
        cv2.waitKey(1)

        # JSON 퍼블리시
        msg        = String()
        msg.data   = json.dumps({
            "mode": self.mode,
            "target": objects[0] if objects else None,
            "objects": objects,
            "detected_count": len(detections),
        }, ensure_ascii=False)
        self.pub_json[self.mode].publish(msg)

    # ------------------------------------------------------------------ #

    def _depth_fallback_pose(self, depth: np.ndarray, mask: np.ndarray):
        """
        FoundationPose 없을 때 또는 실패 시 fallback:
        depth point cloud → centroid + PCA axes → 4×4 matrix.
        (기존 detect_3d_pose.py 방식)
        """
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        rows, cols = np.where(mask > 0)
        z = depth[rows, cols].astype(float) * 0.001
        valid = z > 0
        rows, cols, z = rows[valid], cols[valid], z[valid]
        if len(z) < 10:
            return None

        z_med = np.median(z)
        valid = np.abs(z - z_med) < 0.05
        rows, cols, z = rows[valid], cols[valid], z[valid]
        if len(z) < 10:
            return None

        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)

        centroid = np.median(pts, axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3,  3] = centroid
        return pose

    # ------------------------------------------------------------------ #

    def destroy_node(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = FoundationPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
