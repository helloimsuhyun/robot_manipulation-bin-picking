#!/usr/bin/env python3
"""
python cad_grasp_pose_editor.py \
  --cad /home/choisuhyun/course/robot_manipulation-bin-picking/src/sixd_pose_vision/CAD/cross.stl\
  --yaml /home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml\
  --object cross \
  --output-yaml /home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


def orthonormalize_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt


def validate_T(T: np.ndarray, name: str = "T", atol: float = 1e-3) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)

    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name}: last row must be [0, 0, 0, 1], got {T[3]}")

    R = T[:3, :3]
    det = float(np.linalg.det(R))

    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(f"{name}: det(R) must be close to 1. det={det}")

    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name}: R.T @ R must be close to I.")


def matrix_from_yaml(data: Any, unit: str = "mm") -> np.ndarray:
    T = np.asarray(data, dtype=np.float64).reshape(4, 4).copy()
    T[:3, :3] = orthonormalize_R(T[:3, :3])

    if unit == "m":
        T[:3, 3] *= 1000.0
    elif unit == "mm":
        pass
    else:
        raise ValueError(f"Unsupported unit: {unit}. Use 'mm' or 'm'.")

    validate_T(T, "object_T_grasp")
    return T


def format_matrix_for_yaml(T: np.ndarray, ndigits: int = 6) -> List[List[float]]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    T[:3, :3] = orthonormalize_R(T[:3, :3])
    T[3] = [0.0, 0.0, 0.0, 1.0]

    return [
        [round(float(v), ndigits) for v in row]
        for row in T
    ]


def local_offset_T_mm(axis: str, distance_mm: float) -> np.ndarray:
    axis = str(axis).strip().lower()
    sign = 1.0

    if axis.startswith("-"):
        sign = -1.0
        axis_name = axis[1:]
    elif axis.startswith("+"):
        axis_name = axis[1:]
    else:
        axis_name = axis

    idx_map = {"x": 0, "y": 1, "z": 2}
    if axis_name not in idx_map:
        raise ValueError(f"Unsupported pregrasp_axis: {axis}. Use x, y, z, -x, -y, -z.")

    T = np.eye(4, dtype=np.float64)
    T[idx_map[axis_name], 3] = sign * float(distance_mm)
    return T


def axis_angle_R(axis: str, angle_deg: float) -> np.ndarray:
    axis = axis.lower()
    idx_map = {"x": 0, "y": 1, "z": 2}
    if axis not in idx_map:
        raise ValueError(f"axis must be x, y, or z. got {axis}")

    v = np.zeros(3, dtype=np.float64)
    v[idx_map[axis]] = 1.0

    theta = np.deg2rad(float(angle_deg))
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return orthonormalize_R(R)


def read_grasp_yaml(
    yaml_path: Path,
    object_name: str,
    default_pregrasp_distance_mm: float,
    default_pregrasp_axis: str,
) -> Tuple[Dict[str, Any], np.ndarray, float, str]:
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    global_unit = data.get("unit", "mm")
    global_pre_dist = float(data.get("default_pregrasp_distance_mm", default_pregrasp_distance_mm))
    global_pre_axis = str(data.get("default_pregrasp_axis", default_pregrasp_axis))

    objects = data.get("objects", {})
    obj_data = objects.get(object_name, {}) if isinstance(objects, dict) else {}

    grasp_data = obj_data.get("object_to_grasp", {}) if isinstance(obj_data, dict) else {}
    unit = grasp_data.get("unit", obj_data.get("unit", global_unit)) if isinstance(obj_data, dict) else global_unit

    if isinstance(grasp_data, dict) and "matrix" in grasp_data:
        T_obj_grasp = matrix_from_yaml(grasp_data["matrix"], unit=unit)
        print(f"[INFO] Loaded object_T_grasp for object='{object_name}' from {yaml_path}")
    elif isinstance(obj_data, dict) and "matrix" in obj_data:
        T_obj_grasp = matrix_from_yaml(obj_data["matrix"], unit=unit)
        print(f"[INFO] Loaded legacy matrix for object='{object_name}' from {yaml_path}")
    else:
        T_obj_grasp = np.eye(4, dtype=np.float64)
        print(f"[WARN] No matrix found for object='{object_name}'. Starting with identity.")

    pre_dist = float(obj_data.get("pregrasp_distance_mm", global_pre_dist)) if isinstance(obj_data, dict) else global_pre_dist
    pre_axis = str(obj_data.get("pregrasp_axis", global_pre_axis)) if isinstance(obj_data, dict) else global_pre_axis

    return data, T_obj_grasp, pre_dist, pre_axis


def save_grasp_yaml(
    data: Dict[str, Any],
    output_path: Path,
    object_name: str,
    T_obj_grasp: np.ndarray,
    pregrasp_distance_mm: float,
    pregrasp_axis: str,
) -> None:
    data = copy.deepcopy(data) if isinstance(data, dict) else {}

    if "unit" not in data:
        data["unit"] = "mm"

    if "default_pregrasp_distance_mm" not in data:
        data["default_pregrasp_distance_mm"] = float(pregrasp_distance_mm)

    if "default_pregrasp_axis" not in data:
        data["default_pregrasp_axis"] = str(pregrasp_axis)

    if "objects" not in data or not isinstance(data["objects"], dict):
        data["objects"] = {}

    if object_name not in data["objects"] or not isinstance(data["objects"][object_name], dict):
        data["objects"][object_name] = {}

    obj = data["objects"][object_name]
    obj["object_to_grasp"] = {
        "unit": "mm",
        "matrix": format_matrix_for_yaml(T_obj_grasp),
    }
    obj["pregrasp_distance_mm"] = float(pregrasp_distance_mm)
    obj["pregrasp_axis"] = str(pregrasp_axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"[SAVE] Updated YAML saved to: {output_path}")


def print_matrix_block(T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    print("\nobject_to_grasp:")
    print("  unit: mm")
    print("  matrix:")
    for row in format_matrix_for_yaml(T):
        print(f"    - {row}")
    print("")


def yaw_pitch_roll_deg_from_R_zyx(R: np.ndarray) -> np.ndarray:
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

    return np.rad2deg([rx, ry, rz])


class GraspPoseEditor:
    def __init__(self, args: argparse.Namespace):
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError("Open3D is not installed. Run: pip install open3d") from e

        self.o3d = o3d
        self.args = args

        self.cad_path = Path(args.cad)
        self.yaml_path = Path(args.yaml)
        self.output_yaml = Path(args.output_yaml) if args.output_yaml else self.yaml_path.with_name(
            self.yaml_path.stem + "_updated.yaml"
        )

        self.object_name = args.object
        self.default_pregrasp_distance_mm = float(args.default_pregrasp_distance_mm)
        self.default_pregrasp_axis = str(args.default_pregrasp_axis)

        self.yaml_data, self.T_obj_grasp, self.pre_dist_mm, self.pre_axis = read_grasp_yaml(
            self.yaml_path,
            self.object_name,
            self.default_pregrasp_distance_mm,
            self.default_pregrasp_axis,
        )

        self.T_initial = self.T_obj_grasp.copy()

        self.active_axis = "z"
        self.edit_mode = "local"  # local or object
        self.trans_step_mm = float(args.trans_step_mm)
        self.rot_step_deg = float(args.rot_step_deg)

        self.vis = None
        self.dynamic_geoms = []

    def load_cad_geometry(self):
        if not self.cad_path.exists():
            raise FileNotFoundError(f"CAD file not found: {self.cad_path}")

        o3d = self.o3d
        suffix = self.cad_path.suffix.lower()

        mesh = o3d.io.read_triangle_mesh(str(self.cad_path))
        if mesh is not None and len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.72, 0.72, 0.72])

            if self.args.cad_unit == "m":
                mesh.scale(1000.0, center=(0.0, 0.0, 0.0))
            elif self.args.cad_unit == "mm":
                pass
            else:
                raise ValueError("--cad-unit must be 'mm' or 'm'.")

            print(f"[INFO] Loaded mesh: {self.cad_path}")
            print(f"[INFO] vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
            return mesh

        pcd = o3d.io.read_point_cloud(str(self.cad_path))
        if pcd is not None and len(pcd.points) > 0:
            pcd.paint_uniform_color([0.72, 0.72, 0.72])

            if self.args.cad_unit == "m":
                pcd.scale(1000.0, center=(0.0, 0.0, 0.0))
            elif self.args.cad_unit == "mm":
                pass
            else:
                raise ValueError("--cad-unit must be 'mm' or 'm'.")

            print(f"[INFO] Loaded point cloud: {self.cad_path}")
            print(f"[INFO] points={len(pcd.points)}")
            return pcd

        raise RuntimeError(f"Failed to load CAD/point cloud file: {self.cad_path} ({suffix})")

    def create_line(self, p0, p1, color):
        o3d = self.o3d
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([p0, p1])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([color])
        return line

    def create_dynamic_geoms(self):
        o3d = self.o3d

        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(self.args.frame_size),
            origin=[0.0, 0.0, 0.0],
        )
        grasp_frame.transform(self.T_obj_grasp)

        T_obj_pre = self.T_obj_grasp @ local_offset_T_mm(self.pre_axis, self.pre_dist_mm)
        pregrasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(self.args.frame_size) * 0.75,
            origin=[0.0, 0.0, 0.0],
        )
        pregrasp_frame.transform(T_obj_pre)

        line_to_grasp = self.create_line(
            [0.0, 0.0, 0.0],
            self.T_obj_grasp[:3, 3].tolist(),
            [1.0, 0.6, 0.0],
        )

        line_grasp_to_pre = self.create_line(
            self.T_obj_grasp[:3, 3].tolist(),
            T_obj_pre[:3, 3].tolist(),
            [0.0, 0.8, 1.0],
        )

        return [grasp_frame, pregrasp_frame, line_to_grasp, line_grasp_to_pre]

    def refresh_dynamic_geometry(self):
        if self.vis is None:
            return

        for geom in self.dynamic_geoms:
            self.vis.remove_geometry(geom, reset_bounding_box=False)

        self.dynamic_geoms = self.create_dynamic_geoms()

        for geom in self.dynamic_geoms:
            self.vis.add_geometry(geom, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

    def show_status(self):
        p = self.T_obj_grasp[:3, 3]
        rpy = yaw_pitch_roll_deg_from_R_zyx(self.T_obj_grasp[:3, :3])
        print(
            f"[STATUS] object='{self.object_name}' | "
            f"mode={self.edit_mode} | axis={self.active_axis.upper()} | "
            f"trans_step={self.trans_step_mm:.3f}mm | rot_step={self.rot_step_deg:.3f}deg | "
            f"t=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] mm | "
            f"rpy_zyx_deg=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
        )

    def print_help(self):
        print(
            """
==================== CAD Grasp Pose Editor Controls ====================

View control:
  Mouse drag / wheel       Open3D 기본 시점 회전/확대/이동

Select edit axis:
  X                        active axis = X
  Y                        active axis = Y
  Z                        active axis = Z

Move grasp frame:
  A                        translate -step along active axis
  D                        translate +step along active axis

Rotate grasp frame:
  Q                        rotate -step around active axis
  E                        rotate +step around active axis

Mode:
  G                        toggle edit mode: local axis <-> object/world axis

Step size:
  [                        translation step / 2
  ]                        translation step * 2
  -                        rotation step / 2
  =                        rotation step * 2

YAML / print:
  P                        print current object_to_grasp matrix
  S                        save updated YAML
  R                        reset to initial matrix
  H                        print this help

Meaning:
  object frame             CAD 원점 좌표계
  grasp frame              object_T_grasp
  pregrasp frame           object_T_grasp @ local_offset(pregrasp_axis, distance)

=======================================================================
"""
        )

    def translate(self, sign: float):
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.active_axis]
        step = sign * self.trans_step_mm

        if self.edit_mode == "local":
            direction = self.T_obj_grasp[:3, :3] @ np.eye(3)[:, axis_idx]
        else:
            direction = np.eye(3)[:, axis_idx]

        self.T_obj_grasp[:3, 3] += direction * step
        self.refresh_dynamic_geometry()
        self.show_status()

    def rotate(self, sign: float):
        R_delta = axis_angle_R(self.active_axis, sign * self.rot_step_deg)

        if self.edit_mode == "local":
            self.T_obj_grasp[:3, :3] = self.T_obj_grasp[:3, :3] @ R_delta
        else:
            self.T_obj_grasp[:3, :3] = R_delta @ self.T_obj_grasp[:3, :3]

        self.T_obj_grasp[:3, :3] = orthonormalize_R(self.T_obj_grasp[:3, :3])
        self.refresh_dynamic_geometry()
        self.show_status()

    def register_callbacks(self):
        v = self.vis

        def set_axis(axis):
            def callback(_vis):
                self.active_axis = axis
                self.show_status()
                return False
            return callback

        def cb_translate_minus(_vis):
            self.translate(-1.0)
            return False

        def cb_translate_plus(_vis):
            self.translate(+1.0)
            return False

        def cb_rotate_minus(_vis):
            self.rotate(-1.0)
            return False

        def cb_rotate_plus(_vis):
            self.rotate(+1.0)
            return False

        def cb_toggle_mode(_vis):
            self.edit_mode = "object" if self.edit_mode == "local" else "local"
            self.show_status()
            return False

        def cb_trans_step_down(_vis):
            self.trans_step_mm = max(self.trans_step_mm / 2.0, 0.001)
            self.show_status()
            return False

        def cb_trans_step_up(_vis):
            self.trans_step_mm *= 2.0
            self.show_status()
            return False

        def cb_rot_step_down(_vis):
            self.rot_step_deg = max(self.rot_step_deg / 2.0, 0.001)
            self.show_status()
            return False

        def cb_rot_step_up(_vis):
            self.rot_step_deg *= 2.0
            self.show_status()
            return False

        def cb_print(_vis):
            print_matrix_block(self.T_obj_grasp)
            self.show_status()
            return False

        def cb_save(_vis):
            save_grasp_yaml(
                data=self.yaml_data,
                output_path=self.output_yaml,
                object_name=self.object_name,
                T_obj_grasp=self.T_obj_grasp,
                pregrasp_distance_mm=self.pre_dist_mm,
                pregrasp_axis=self.pre_axis,
            )
            print_matrix_block(self.T_obj_grasp)
            return False

        def cb_reset(_vis):
            self.T_obj_grasp = self.T_initial.copy()
            self.refresh_dynamic_geometry()
            self.show_status()
            return False

        def cb_help(_vis):
            self.print_help()
            return False

        v.register_key_callback(ord("X"), set_axis("x"))
        v.register_key_callback(ord("Y"), set_axis("y"))
        v.register_key_callback(ord("Z"), set_axis("z"))

        v.register_key_callback(ord("A"), cb_translate_minus)
        v.register_key_callback(ord("D"), cb_translate_plus)

        v.register_key_callback(ord("Q"), cb_rotate_minus)
        v.register_key_callback(ord("E"), cb_rotate_plus)

        v.register_key_callback(ord("G"), cb_toggle_mode)

        v.register_key_callback(ord("["), cb_trans_step_down)
        v.register_key_callback(ord("]"), cb_trans_step_up)
        v.register_key_callback(ord("-"), cb_rot_step_down)
        v.register_key_callback(ord("="), cb_rot_step_up)

        v.register_key_callback(ord("P"), cb_print)
        v.register_key_callback(ord("S"), cb_save)
        v.register_key_callback(ord("R"), cb_reset)
        v.register_key_callback(ord("H"), cb_help)

    def run(self):
        o3d = self.o3d

        cad_geom = self.load_cad_geometry()

        object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(self.args.frame_size),
            origin=[0.0, 0.0, 0.0],
        )

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=f"CAD Grasp Pose Editor - {self.object_name}",
            width=int(self.args.width),
            height=int(self.args.height),
        )

        self.vis.add_geometry(cad_geom)
        self.vis.add_geometry(object_frame)

        self.dynamic_geoms = self.create_dynamic_geoms()
        for geom in self.dynamic_geoms:
            self.vis.add_geometry(geom)

        render_option = self.vis.get_render_option()
        render_option.background_color = np.asarray([0.05, 0.05, 0.05])
        render_option.mesh_show_back_face = True
        render_option.point_size = 3.0

        self.register_callbacks()

        self.print_help()
        print_matrix_block(self.T_obj_grasp)
        self.show_status()

        self.vis.run()
        self.vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CAD grasp pose editor for object_grasp.yaml")

    parser.add_argument("--cad", required=True, help="CAD path: .stl, .obj, .ply, etc.")
    parser.add_argument("--yaml", required=True, help="object_grasp.yaml path")
    parser.add_argument("--object", required=True, help="object name in YAML, e.g. cylinder, hole, cross")

    parser.add_argument("--output-yaml", default="", help="output YAML path. Default: <input>_updated.yaml")
    parser.add_argument("--cad-unit", default="mm", choices=["mm", "m"], help="CAD coordinate unit. Default: mm")

    parser.add_argument("--frame-size", type=float, default=50.0, help="coordinate frame size in mm")
    parser.add_argument("--trans-step-mm", type=float, default=5.0, help="initial translation step in mm")
    parser.add_argument("--rot-step-deg", type=float, default=5.0, help="initial rotation step in degree")

    parser.add_argument("--default-pregrasp-distance-mm", type=float, default=80.0)
    parser.add_argument("--default-pregrasp-axis", default="-z")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)

    return parser.parse_args()


def main():
    args = parse_args()
    editor = GraspPoseEditor(args)
    editor.run()


if __name__ == "__main__":
    main()