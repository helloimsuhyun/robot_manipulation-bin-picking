#!/usr/bin/env python3
"""
RealSense keyboard capture ROS2 node.

Purpose:
  - Directly capture aligned RealSense RGB-D frames.
  - Show live color/depth preview.
  - Press keyboard keys to save synchronized color/depth data.

Keys:
  s or Space : save current frame
  q or ESC   : quit preview / shutdown node
  h          : print help

Saved per capture:
  capture_XXXXXX_color.png        BGR color image
  capture_XXXXXX_depth_raw.png    uint16 raw depth image
  capture_XXXXXX_depth_m.npy      float32 depth in meters
  capture_XXXXXX_depth_vis.png    depth colormap visualization
  capture_XXXXXX_intrinsics.json  color camera intrinsics + depth scale
  capture_XXXXXX_points.ply       optional full RGB-D point cloud, if save_pointcloud=true

Example:
  cd ~/course/robot_manipulation-bin-picking
  colcon build --symlink-install --packages-select sixd_pose_vision
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  ros2 run sixd_pose_vision realsense_keyboard_capture_node

Optional parameters:
  ros2 run sixd_pose_vision realsense_keyboard_capture_node --ros-args \
    -p save_dir:=/home/chu/realsense_captures \
    -p color_width:=848 -p color_height:=480 -p fps:=30 \
    -p save_pointcloud:=true
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node


class RealSenseKeyboardCaptureNode(Node):
    def __init__(self):
        super().__init__("realsense_keyboard_capture_node")

        # RealSense params
        self.declare_parameter("color_width", 848)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("enable_align", True)

        # Save params
        self.declare_parameter("save_dir", "realsense_captures")
        self.declare_parameter("save_pointcloud", False)
        self.declare_parameter("pointcloud_stride", 2)  # 1 saves all valid pixels; 2 is lighter
        self.declare_parameter("depth_vis_min_m", 0.25)
        self.declare_parameter("depth_vis_max_m", 1.20)

        self.width = int(self.get_parameter("color_width").value)
        self.height = int(self.get_parameter("color_height").value)
        self.fps = int(self.get_parameter("fps").value)
        self.enable_align = bool(self.get_parameter("enable_align").value)

        save_dir_param = Path(str(self.get_parameter("save_dir").value)).expanduser()
        self.save_dir = save_dir_param if save_dir_param.is_absolute() else (Path.cwd() / save_dir_param)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_pointcloud = bool(self.get_parameter("save_pointcloud").value)
        self.pointcloud_stride = max(1, int(self.get_parameter("pointcloud_stride").value))

        self.capture_idx = self._find_next_index()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.latest_depth_m: Optional[np.ndarray] = None

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if self.enable_align else None

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = rs.video_stream_profile(color_stream).get_intrinsics()
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())

        self.window_name = "RealSense Capture | s/space=save, q/esc=quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(1.0 / max(float(self.fps), 1.0), self.timer_callback)

        self.get_logger().info(
            f"RealSenseKeyboardCaptureNode ready | {self.width}x{self.height}@{self.fps} "
            f"align={self.enable_align} depth_scale={self.depth_scale:.6f} "
            f"save_dir={self.save_dir} save_pointcloud={self.save_pointcloud}"
        )
        self.print_help()

    def _find_next_index(self) -> int:
        existing = sorted(self.save_dir.glob("capture_*_color.png"))
        max_idx = -1
        for p in existing:
            try:
                # capture_000012_color.png -> 12
                idx = int(p.name.split("_")[1])
                max_idx = max(max_idx, idx)
            except Exception:
                pass
        return max_idx + 1

    def print_help(self):
        self.get_logger().info(
            "Keys: s or SPACE = save RGB-D frame | q or ESC = quit | h = help"
        )

    def read_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        color = np.asanyarray(color_frame.get_data())  # BGR8
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        return color, depth_raw

    def make_depth_vis(self, depth_m: np.ndarray) -> np.ndarray:
        vis_min = float(self.get_parameter("depth_vis_min_m").value)
        vis_max = float(self.get_parameter("depth_vis_max_m").value)
        if vis_max <= vis_min:
            vis_max = vis_min + 1e-3

        valid = depth_m > 0.0
        norm = np.clip((depth_m - vis_min) / (vis_max - vis_min), 0.0, 1.0)
        depth_u8 = (norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_color[~valid] = (0, 0, 0)
        return depth_color

    def make_preview(self, color: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        depth_vis = self.make_depth_vis(depth_m)
        preview = np.hstack([color, depth_vis])

        text1 = "s/SPACE: save | q/ESC: quit | h: help"
        text2 = f"idx={self.capture_idx:06d} | save_dir={self.save_dir}"
        cv2.putText(preview, text1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(preview, text2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return preview

    def timer_callback(self):
        try:
            color, depth_raw = self.read_frame()
            if color is None or depth_raw is None:
                return

            depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)

            self.latest_color = color
            self.latest_depth_raw = depth_raw
            self.latest_depth_m = depth_m

            preview = self.make_preview(color, depth_m)
            cv2.imshow(self.window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord(" ")):
                self.save_current_frame()
            elif key in (ord("q"), 27):
                self.get_logger().info("Quit key received. Shutting down.")
                rclpy.shutdown()
            elif key == ord("h"):
                self.print_help()

        except Exception as e:
            self.get_logger().warn(f"timer_callback failed: {e}")

    def save_current_frame(self):
        if self.latest_color is None or self.latest_depth_raw is None or self.latest_depth_m is None:
            self.get_logger().warn("No frame available to save yet.")
            return

        idx = self.capture_idx
        stem = self.save_dir / f"capture_{idx:06d}"

        color_path = Path(str(stem) + "_color.png")
        depth_raw_path = Path(str(stem) + "_depth_raw.png")
        depth_m_path = Path(str(stem) + "_depth_m.npy")
        depth_vis_path = Path(str(stem) + "_depth_vis.png")
        intrinsics_path = Path(str(stem) + "_intrinsics.json")
        ply_path = Path(str(stem) + "_points.ply")

        color = self.latest_color.copy()
        depth_raw = self.latest_depth_raw.copy()
        depth_m = self.latest_depth_m.copy()
        depth_vis = self.make_depth_vis(depth_m)

        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(depth_raw_path), depth_raw)
        np.save(str(depth_m_path), depth_m)
        cv2.imwrite(str(depth_vis_path), depth_vis)

        meta = {
            "index": int(idx),
            "width": int(self.width),
            "height": int(self.height),
            "fps": int(self.fps),
            "align_to_color": bool(self.enable_align),
            "depth_scale": float(self.depth_scale),
            "intrinsics": {
                "width": int(self.intrinsics.width),
                "height": int(self.intrinsics.height),
                "fx": float(self.intrinsics.fx),
                "fy": float(self.intrinsics.fy),
                "ppx": float(self.intrinsics.ppx),
                "ppy": float(self.intrinsics.ppy),
                "model": str(self.intrinsics.model),
                "coeffs": [float(v) for v in self.intrinsics.coeffs],
            },
            "files": {
                "color_png": str(color_path),
                "depth_raw_png": str(depth_raw_path),
                "depth_m_npy": str(depth_m_path),
                "depth_vis_png": str(depth_vis_path),
            },
        }

        if self.save_pointcloud:
            n = self.save_point_cloud_ply(ply_path, color, depth_m)
            meta["files"]["pointcloud_ply"] = str(ply_path)
            meta["pointcloud_points"] = int(n)

        with open(intrinsics_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.get_logger().info(
            f"[CAPTURED] idx={idx:06d} color={color_path} depth_raw={depth_raw_path} "
            f"depth_m={depth_m_path} meta={intrinsics_path}"
        )

        self.capture_idx += 1

    def save_point_cloud_ply(self, path: Path, color_bgr: np.ndarray, depth_m: np.ndarray) -> int:
        h, w = depth_m.shape[:2]
        stride = self.pointcloud_stride

        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_m[ys, xs]
        valid = z > 0.0

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        if len(z) == 0:
            return 0

        x = (xs - float(self.intrinsics.ppx)) * z / float(self.intrinsics.fx)
        y = (ys - float(self.intrinsics.ppy)) * z / float(self.intrinsics.fy)

        pts = np.stack([x, y, z], axis=1)
        colors_bgr = color_bgr[ys.astype(np.int32), xs.astype(np.int32)]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
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

            for p, c in zip(pts, colors_bgr):
                b, g, r = int(c[0]), int(c[1]), int(c[2])
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")

        return int(len(pts))

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseKeyboardCaptureNode()
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