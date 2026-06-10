"""
Build / run example:

cd ~/course/robot_manipulation-bin-picking

colcon build --symlink-install --packages-select sixd_pose_vision calib

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch calib vision_transform_test_bringup.launch.py

# Keyboard trigger는 다른 터미널에서 따로 실행:
# ros2 run sixd_pose_vision keyboard_object_trigger_node --ros-args \
#   -p trigger_topic:=/manipulation/object_6d_trigger \
#   -p detect_mode_topic:=/detect_mode
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ============================================================
    # PC 변경 시 여기 경로만 먼저 확인
    # ============================================================

    foundationpose_repo_path = "/home/choisuhyun/FoundationPose"

    handeye_result_path = (
        "/home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/handeye_capture_rs/handeye_result.json"
    )

    object_grasp_yaml_path = (
        "/home/choisuhyun/course/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml"
    )

    robot_ip = "192.168.1.10"
    use_robot = True

    # ============================================================
    # Topic names
    # ============================================================

    object_topic = "/object_poses"
    insert_topic = "/insert_poses"
    detect_mode_topic = "/detect_mode"
    object_6d_trigger_topic = "/manipulation/object_6d_trigger"

    sixd_pose_share = FindPackageShare("sixd_pose_vision")

    # ============================================================
    # 1) Vision node: YOLO/SAM/FoundationPose mixed perception
    # ============================================================

    mixed_pose_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                sixd_pose_share,
                "launch",
                "mixed_pose_vision.launch.py",
            ])
        ),
        launch_arguments={
            "foundationpose_repo_path": foundationpose_repo_path,

            "default_mode": "object",
            "enable_visualization": "true",
            "conf_thresh": "0.4",

            "color_width": "848",
            "color_height": "480",
            "fps": "30",
            "frame_id": "camera_color_optical_frame",

            "fp_register_iter": "5",
            "fp_track_iter": "2",
            "fp_track_loss_thr": "0.2",
            "fp_trigger_track_frames": "10",
            "fp_trigger_track_use_new_frames": "true",
            "fp_debug": "0",

            "object_topic": object_topic,
            "insert_topic": insert_topic,
            "detect_mode_topic": detect_mode_topic,
            "object_trigger_topic": object_6d_trigger_topic,
        }.items(),
    )

    # ============================================================
    # 2) Transform test node:
    #    /object_poses 수신 -> 현재 TCP 직접 쿼리 -> pose6 시각화
    # ============================================================

    sixd_pose_transform_test_node = Node(
        package="calib",
        executable="sixd_pose_transform_test",
        name="sixd_pose_transform_test_node",
        output="screen",
        parameters=[{
            "robot_ip": robot_ip,
            "use_robot": use_robot,

            "object_topic": object_topic,
            "handeye_result_path": handeye_result_path,
            "object_grasp_yaml_path": object_grasp_yaml_path,

            "min_confidence": 0.3,

            "canonicalize_object_axes": True,
            "canonicalize_z_flip_margin": 0.05,

            "canonicalize_xy_flatter_as_x": True,
            "canonicalize_xy_swap_margin": 0.03,
            "canonicalize_xy_max_flatness": 0.85,

            "reference_last_joint_deg": 34.16,
            "last_joint_limit_delta_deg": 95.0,

            "visualize_axes_length_mm": 50.0,
            "visualize_approach_length_mm": 80.0,
            "visualize_save_dir": "",
        }],
    )

    return LaunchDescription([
        mixed_pose_vision_launch,
        sixd_pose_transform_test_node,
    ])