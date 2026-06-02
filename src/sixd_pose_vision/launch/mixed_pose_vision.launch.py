from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    pkg_share_dir = launch_dir.parent

    default_foundationpose_repo_path = "/home/choisuhyun/course/robot_manipulation-bin-picking/FoundationPose"

    default_cad_dir = str(pkg_share_dir / "CAD")
    default_template_dir = str(pkg_share_dir / "templates")

    default_object_yolo_path = str(pkg_share_dir / "weights" / "object" / "best.pt")
    default_insert_yolo_path = str(pkg_share_dir / "weights" / "insert" / "insert_best.pt")

    return LaunchDescription([
        DeclareLaunchArgument("foundationpose_repo_path", default_value=default_foundationpose_repo_path),
        DeclareLaunchArgument("cad_dir", default_value=default_cad_dir),
        DeclareLaunchArgument("template_dir", default_value=default_template_dir),
        DeclareLaunchArgument("object_yolo_path", default_value=default_object_yolo_path),
        DeclareLaunchArgument("insert_yolo_path", default_value=default_insert_yolo_path),

        DeclareLaunchArgument("mesh_scale", default_value="0.001"),
        DeclareLaunchArgument("default_mode", default_value="object"),
        DeclareLaunchArgument("enable_visualization", default_value="true"),
        DeclareLaunchArgument("conf_thresh", default_value="0.4"),

        # FoundationPose
        DeclareLaunchArgument("fp_register_iter", default_value="2"),
        DeclareLaunchArgument("fp_track_iter", default_value="2"),
        DeclareLaunchArgument("fp_track_loss_thr", default_value="0.2"),
        DeclareLaunchArgument("fp_debug", default_value="0"),
        DeclareLaunchArgument(
            "fp_debug_dir",
            default_value="/home/choisuhyun/course/robot_manipulation-bin-picking/FoundationPose/debug_ros",
        ),

        # 현재 object mode 코드는 trigger마다 register-only로 돌기 때문에 false가 더 명확함
        DeclareLaunchArgument("fp_use_tracking", default_value="false"),

        # Topics
        DeclareLaunchArgument("object_topic", default_value="/object_poses"),
        DeclareLaunchArgument("insert_topic", default_value="/insert_poses"),
        DeclareLaunchArgument("object_pose_topic", default_value="/object_pose_stamped"),
        DeclareLaunchArgument("insert_pose_topic", default_value="/insert_pose_stamped"),
        DeclareLaunchArgument("detect_mode_topic", default_value="/detect_mode"),
        DeclareLaunchArgument("object_trigger_topic", default_value="/object_6d_trigger"),
        DeclareLaunchArgument("frame_id", default_value="camera_color_optical_frame"),

        Node(
            package="sixd_pose_vision",
            executable="mixed_pose_vision_node",
            name="mixed_pose_vision_node",
            output="screen",
            parameters=[{
                "foundationpose_repo_path": LaunchConfiguration("foundationpose_repo_path"),
                "cad_dir": LaunchConfiguration("cad_dir"),
                "template_dir": LaunchConfiguration("template_dir"),
                "object_yolo_path": LaunchConfiguration("object_yolo_path"),
                "insert_yolo_path": LaunchConfiguration("insert_yolo_path"),

                "mesh_scale": ParameterValue(LaunchConfiguration("mesh_scale"), value_type=float),
                "default_mode": LaunchConfiguration("default_mode"),
                "enable_visualization": ParameterValue(LaunchConfiguration("enable_visualization"), value_type=bool),
                "conf_thresh": ParameterValue(LaunchConfiguration("conf_thresh"), value_type=float),

                "fp_register_iter": ParameterValue(LaunchConfiguration("fp_register_iter"), value_type=int),
                "fp_track_iter": ParameterValue(LaunchConfiguration("fp_track_iter"), value_type=int),
                "fp_track_loss_thr": ParameterValue(LaunchConfiguration("fp_track_loss_thr"), value_type=float),
                "fp_debug": ParameterValue(LaunchConfiguration("fp_debug"), value_type=int),
                "fp_debug_dir": LaunchConfiguration("fp_debug_dir"),
                "fp_use_tracking": ParameterValue(LaunchConfiguration("fp_use_tracking"), value_type=bool),

                "object_topic": LaunchConfiguration("object_topic"),
                "insert_topic": LaunchConfiguration("insert_topic"),
                "object_pose_topic": LaunchConfiguration("object_pose_topic"),
                "insert_pose_topic": LaunchConfiguration("insert_pose_topic"),
                "detect_mode_topic": LaunchConfiguration("detect_mode_topic"),
                "object_trigger_topic": LaunchConfiguration("object_trigger_topic"),
                "frame_id": LaunchConfiguration("frame_id"),
            }],
        ),
    ])