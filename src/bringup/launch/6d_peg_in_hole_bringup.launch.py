"""
Build / run example:

colcon build --symlink-install --packages-select sixd_pose_vision calib control bringup
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch bringup 6d_peg_in_hole_bringup.launch.py

source /opt/ros/humble/setup.bash
python3 src/robot_ex_2026/robot_ex_2026/grip_current.py

source /opt/ros/humble/setup.bash
ros2 topic pub --once /manual_continue std_msgs/msg/Empty "{}"

"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ============================================================
    # PC 변경 시 여기 3개 경로만 먼저 확인 **********************************************************

    foundationpose_repo_path = "/home/chu/FoundationPose"

    handeye_result_path = (
        "/home/chu/robot_manipulation-bin-picking/src/calib/config/handeye_capture_rs/handeye_result.json"
    )

    object_grasp_yaml_path = (
        "/home/chu/robot_manipulation-bin-picking/src/calib/config/object_grasp.yaml"
    )

    # ============================================================
    # Topic names

    object_topic = "/object_poses"
    insert_topic = "/insert_poses"
    detect_mode_topic = "/detect_mode"

    peg_trigger_topic = "/manipulation/trigger_peg"
    hole_trigger_topic = "/manipulation/trigger_hole"
    object_6d_trigger_topic = "/manipulation/object_6d_trigger"

    peg_output_topic = "/vision/peg_targets"
    hole_output_topic = "/vision/hole_targets"

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
            "fp_trigger_track_frames": "10",
            "fp_trigger_track_use_new_frames": "true",
            "fp_debug": "0",

            "object_topic": object_topic,
            "insert_topic": insert_topic,
            "detect_mode_topic": detect_mode_topic,
            "object_trigger_topic": object_6d_trigger_topic,

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

            # object_to_center 적용 후 centered object +Z가 world +Z에 더 가깝도록 보정
            "canonicalize_object_axes": True,
            "canonicalize_z_flip_margin": 0.05,
            #    X/Y 중 world XY 평면에 더 평행한 축을 centered object +X로 선택
            "canonicalize_xy_flatter_as_x": True,
            "canonicalize_xy_swap_margin": 0.03,
            "canonicalize_xy_max_flatness": 0.85,

            # 3) YAML symmetry.yaw_candidates_deg 후보 중 RB5 마지막 joint 기준으로 선택
            #    peg_camera_joint 마지막 joint J5=34.16 deg 기준
            "reference_last_joint_deg": 34.16,
            "last_joint_limit_delta_deg": 95.0,

            "min_confidence": 0.3,

            "object_topic": object_topic,
            "insert_topic": insert_topic,
            "detect_mode_topic": detect_mode_topic,

            "peg_trigger_topic": peg_trigger_topic,
            "hole_trigger_topic": hole_trigger_topic,
            "object_6d_trigger_topic": object_6d_trigger_topic,

            "peg_output_topic": peg_output_topic,
            "hole_output_topic": hole_output_topic,

            "detect_mode_settle_sec": 0.5,

            # 실제 제어 노드 구조는 유지하고, publish 직후 preview만 띄움
            "visualize_pose6_target": False,
            "visualize_axes_length_mm": 50.0,
            "visualize_approach_length_mm": 80.0,
            "visualize_blocking": False,
            "visualize_save_dir": "",
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
            'peg_camera_joint': [9.68, 8.52, 55.63, 25.85, 90.0, 35.39],
            'hole_camera_joint': [-173.76, 8.07, 56.16, 25.78, 90.0, 38.83],
            "peg_return_mid_joint": [-47.0, 2.78, 79.15, 8.07, 90.0, 34.16],

            "pick_down_target_z_mm": 69.83,
            "pick_approach_offset_z_mm": 30.0,
            "pick_up_target_z_mm": 110.0,
            "pick_approach_above_peg_z_mm": 40.0,
            "pick_grasp_above_peg_z_mm": 30.0,
            "pick_lift_above_peg_z_mm": 80.0,

            "place_approach_target_z_mm": 108.0,
            "place_down_target_z_mm": 81.0,
            "place_up_target_z_mm": 110.0,

            "move_j_speed": 90.0,
            "move_j_acc": 90.0,
            "move_l_speed": 60.0,
            "move_l_acc": 120.0,
            "approach_move_l_speed": 150.0,
            "approach_move_l_acc": 150.0,
            "descend_move_l_speed": 20.0,
            "descend_move_l_acc": 40.0,

            "peg_targets_topic": peg_output_topic,
            "hole_targets_topic": hole_output_topic,
            "trigger_peg_topic": peg_trigger_topic,
            "trigger_hole_topic": hole_trigger_topic,

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