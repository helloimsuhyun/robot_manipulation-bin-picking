"""
cd ~/course/robot_manipulation-bin-picking

rm -rf build/ install/ log/

colcon build --symlink-install --packages-select sixd_pose_vision calib control bringup

source install/setup.bash

ros2 launch bringup 6d_peg_in_hole_bringup.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ============================================================
    # 핸드아이 , 물체 파지좌표 , FD pose repo 절대경로 PC 변경시 같이 확인 필요 **************************
    foundationpose_repo_path = (
        "/home/chu/FoundationPose"
    )

    handeye_result_path = (
        "/home/chu/robot_manipulation-bin-picking/src/calib/config/handeye_capture_rs/handeye_result.json"
    )

    object_grasp_yaml_path = (
        "/home/chu/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml"
    )
    sixd_pose_share = FindPackageShare("sixd_pose_vision")

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

            "fp_register_iter": "5",
            "fp_track_iter": "2",
            "fp_track_loss_thr": "0.2",
            "fp_debug": "0",
            "fp_use_tracking": "false",

            "object_topic": "/object_poses",
            "insert_topic": "/insert_poses",
            "detect_mode_topic": "/detect_mode",
            "object_trigger_topic": "/object_6d_trigger",
            "frame_id": "camera_color_optical_frame",
        }.items(),
    )

    sixd_pose_transform_node = Node(
        package="calib",
        executable="sixd_pose_transform_node",
        name="sixd_pose_transform_node",
        output="screen",
        parameters=[{
            "handeye_result_path": handeye_result_path,
            "object_grasp_yaml_path": object_grasp_yaml_path,

            # 제어부가 base_T_object를 받아서 직접 TCP target frame을 만든다.
            # 따라서 여기서는 grasp frame이 아니라 object frame을 publish한다.
            "peg_target_pose_mode": "grasp",

            # object frame 축을 제어부에서 그대로 사용해야 하므로 자동 z-flip/xy-swap을 끈다.
            "canonicalize_object_axes": False,
            "canonicalize_grasp_axes": True,
            "canonicalize_z_flip_margin": 0.05,
            "canonicalize_xy_down_margin": 0.05,

            "min_confidence": 0.3,

            "object_topic": "/object_poses",
            "insert_topic": "/insert_poses",
            "detect_mode_topic": "/detect_mode",

            "peg_trigger_topic": "/manipulation/trigger_peg",
            "hole_trigger_topic": "/manipulation/trigger_hole",

            "object_6d_trigger_topic": "/object_6d_trigger",

            "peg_output_topic": "/vision/peg_targets",
            "hole_output_topic": "/vision/hole_targets",

            "detect_mode_settle_sec": 0.5,
        }],
    )

    peg_in_hole_controller_node = Node(
        package="control",
        executable="peg_in_hole_controller",
        name="peg_in_hole_controller",
        output="screen",
        parameters=[{
            "robot_ip": "192.168.1.10",
            "use_simulation_mode": False,

            "gripper_topic": "/grip_state",
            "grip_open": 1,
            "grip_close": 0,
            "grip_stop": 2,

            "home_joint": [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0],
            "peg_camera_joint": [10.87, 2.78, 79.15, 8.07, 90.0, 34.16],
            "hole_camera_joint": [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16],
            "peg_return_mid_joint": [-47.0, 2.78, 79.15, 8.07, 90.0, 34.16],

            "pick_down_target_z_mm": 69.83,
            "pick_approach_offset_z_mm": 30.0,
            "pick_up_target_z_mm": 110.0,
            "pick_approach_above_peg_z_mm": 40.0,
            "pick_grasp_above_peg_z_mm": 30.0,
            "pick_lift_above_peg_z_mm": 80.0,

            "place_approach_target_z_mm": 108.0,
            "place_down_target_z_mm": 98.0,
            "place_up_target_z_mm": 110.0,

            "move_j_speed": 60.0,
            "move_j_acc": 60.0,
            "move_l_speed": 60.0,
            "move_l_acc": 120.0,
            "approach_move_l_speed": 60.0,
            "approach_move_l_acc": 120.0,
            "descend_move_l_speed": 20.0,
            "descend_move_l_acc": 40.0,

            "peg_targets_topic": "/vision/peg_targets",
            "hole_targets_topic": "/vision/hole_targets",
            "trigger_peg_topic": "/manipulation/trigger_peg",
            "trigger_hole_topic": "/manipulation/trigger_hole",

            "camera_settle_sec": 0.5,
            "use_6d_peg_interface": True,
            "vision_wait_timeout_sec": 3.0,
            "grasp_wait_sec": 1.0,
            "release_wait_sec": 1.0,
            "move_start_timeout_sec": 3.0,
        }],
    )

    return LaunchDescription([
        mixed_pose_vision_launch,
        sixd_pose_transform_node,
        TimerAction(
            period=5.0,
            actions=[peg_in_hole_controller_node],
        ),
    ])
