from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

"""
Build / run example:

cd ~/course/robot_manipulation-bin-picking

rm -rf build/sixd_pose_vision install/sixd_pose_vision

source /opt/ros/humble/setup.bash
conda activate cource

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/choisuhyun/FoundationPose:${PYTHONPATH:-}"

python -m colcon build --symlink-install --packages-select sixd_pose_vision
source install/setup.bash

ros2 launch sixd_pose_vision mixed_pose_vision.launch.py
"""


def generate_launch_description():
    pkg_share = FindPackageShare("sixd_pose_vision")

    # 단독 실행 시 사용할 기본 FoundationPose 경로
    default_foundationpose_repo_path = "/home/choisuhyun/FoundationPose"

    return LaunchDescription([
        # ============================================================
        # Path arguments
        # ============================================================

        DeclareLaunchArgument(
            "foundationpose_repo_path",
            default_value=default_foundationpose_repo_path,
        ),

        DeclareLaunchArgument(
            "cad_dir",
            default_value=PathJoinSubstitution([
                pkg_share,
                "CAD",
            ]),
        ),

        DeclareLaunchArgument(
            "template_dir",
            default_value=PathJoinSubstitution([
                pkg_share,
                "templates",
            ]),
        ),

        DeclareLaunchArgument(
            "object_yolo_path",
            default_value=PathJoinSubstitution([
                pkg_share,
                "weights",
                "object",
                "best.pt",
            ]),
        ),

        DeclareLaunchArgument(
            "insert_yolo_path",
            default_value=PathJoinSubstitution([
                pkg_share,
                "weights",
                "insert",
                "insert_best.pt",
            ]),
        ),

        # CAD가 mm 단위면 0.001, m 단위면 1.0
        DeclareLaunchArgument("mesh_scale", default_value="0.001"),

        # ============================================================
        # Runtime arguments
        # ============================================================

        DeclareLaunchArgument("default_mode", default_value="object"),
        DeclareLaunchArgument("enable_visualization", default_value="true"),
        DeclareLaunchArgument("conf_thresh", default_value="0.4"),

        DeclareLaunchArgument("color_width", default_value="848"),
        DeclareLaunchArgument("color_height", default_value="480"),
        DeclareLaunchArgument("fps", default_value="30"),

        DeclareLaunchArgument("frame_id", default_value="camera_color_optical_frame"),

        # ============================================================
        # FoundationPose arguments
        # ============================================================

        # trigger마다 fresh register 1회
        DeclareLaunchArgument("fp_register_iter", default_value="5"),

        # register 이후 짧은 track_one refinement
        DeclareLaunchArgument("fp_track_iter", default_value="2"),
        DeclareLaunchArgument("fp_track_loss_thr", default_value="0.2"),

        # short tracking refinement
        DeclareLaunchArgument("fp_trigger_track_frames", default_value="10"),
        DeclareLaunchArgument("fp_trigger_track_use_new_frames", default_value="true"),

        # 디버그 저장 끄기
        DeclareLaunchArgument("fp_debug", default_value="0"),

        DeclareLaunchArgument(
            "fp_debug_dir",
            default_value=PathJoinSubstitution([
                LaunchConfiguration("foundationpose_repo_path"),
                "debug_ros",
            ]),
        ),

        # ============================================================
        # Empty-space arguments
        # ============================================================

        # object pose JSON 안에 bundled empty_space로 들어가지만,
        # 디버그/시각화용으로 /empty_space_candidates topic도 유지.
        DeclareLaunchArgument("empty_space_topic", default_value="/empty_space_candidates"),
        DeclareLaunchArgument("empty_space_enable", default_value="true"),

        # 이미지 grid 후보 간격.
        # 작을수록 후보가 촘촘하지만 연산량 증가.
        DeclareLaunchArgument("empty_grid_step_px", default_value="40"),

        # vision node가 publish/object JSON에 포함할 최대 후보 수.
        DeclareLaunchArgument("empty_max_candidates", default_value="30"),

        # Empty-space 탐색 ROI.
        # x_max/y_max가 -1이면 width-right_margin, height-bottom_margin 사용.
        DeclareLaunchArgument("empty_roi_x_min", default_value="70"),
        DeclareLaunchArgument("empty_roi_y_min", default_value="60"),
        DeclareLaunchArgument("empty_roi_x_max", default_value="-1"),
        DeclareLaunchArgument("empty_roi_y_max", default_value="-1"),
        DeclareLaunchArgument("empty_roi_right_margin", default_value="70"),
        DeclareLaunchArgument("empty_roi_bottom_margin", default_value="50"),

        # object mask 주변 제외 영역.
        # 픽셀 단위 dilation. 너무 작으면 물체 옆 후보가 살아남고,
        # 너무 크면 후보가 과하게 줄어듦.
        DeclareLaunchArgument("empty_mask_dilate_px", default_value="22"),

        # 후보점 주변 patch depth 검증.
        DeclareLaunchArgument("empty_depth_patch_radius_px", default_value="6"),
        DeclareLaunchArgument("empty_depth_valid_ratio_min", default_value="0.50"),
        DeclareLaunchArgument("empty_depth_spread_max_m", default_value="0.030"),

        # OpenCV 시각화 유지 시간.
        DeclareLaunchArgument("empty_space_vis_hold_sec", default_value="2.0"),

        # ============================================================
        # Priority debug arguments
        # ============================================================

        DeclareLaunchArgument("priority_debug_save", default_value="true"),
        DeclareLaunchArgument("priority_debug_dir", default_value="debug_priority"),

        # ============================================================
        # Topic arguments
        # ============================================================

        DeclareLaunchArgument("object_topic", default_value="/object_poses"),
        DeclareLaunchArgument("insert_topic", default_value="/insert_poses"),

        DeclareLaunchArgument("object_pose_topic", default_value="/object_pose_stamped"),
        DeclareLaunchArgument("insert_pose_topic", default_value="/insert_pose_stamped"),

        DeclareLaunchArgument("detect_mode_topic", default_value="/detect_mode"),

        # transform node와 맞춘 기본값
        DeclareLaunchArgument(
            "object_trigger_topic",
            default_value="/manipulation/object_6d_trigger",
        ),

        # ============================================================
        # Node
        # ============================================================

        Node(
            package="sixd_pose_vision",
            executable="mixed_pose_vision_node",
            name="mixed_pose_vision_node",
            output="screen",
            parameters=[{
                # ----------------------------
                # Paths
                # ----------------------------
                "foundationpose_repo_path": LaunchConfiguration("foundationpose_repo_path"),
                "cad_dir": LaunchConfiguration("cad_dir"),
                "template_dir": LaunchConfiguration("template_dir"),
                "object_yolo_path": LaunchConfiguration("object_yolo_path"),
                "insert_yolo_path": LaunchConfiguration("insert_yolo_path"),

                "mesh_scale": ParameterValue(
                    LaunchConfiguration("mesh_scale"),
                    value_type=float,
                ),

                # ----------------------------
                # Runtime
                # ----------------------------
                "default_mode": LaunchConfiguration("default_mode"),

                "enable_visualization": ParameterValue(
                    LaunchConfiguration("enable_visualization"),
                    value_type=bool,
                ),

                "conf_thresh": ParameterValue(
                    LaunchConfiguration("conf_thresh"),
                    value_type=float,
                ),

                "color_width": ParameterValue(
                    LaunchConfiguration("color_width"),
                    value_type=int,
                ),

                "color_height": ParameterValue(
                    LaunchConfiguration("color_height"),
                    value_type=int,
                ),

                "fps": ParameterValue(
                    LaunchConfiguration("fps"),
                    value_type=int,
                ),

                "frame_id": LaunchConfiguration("frame_id"),

                # ----------------------------
                # FoundationPose
                # ----------------------------
                "fp_register_iter": ParameterValue(
                    LaunchConfiguration("fp_register_iter"),
                    value_type=int,
                ),

                "fp_track_iter": ParameterValue(
                    LaunchConfiguration("fp_track_iter"),
                    value_type=int,
                ),

                "fp_track_loss_thr": ParameterValue(
                    LaunchConfiguration("fp_track_loss_thr"),
                    value_type=float,
                ),

                "fp_trigger_track_frames": ParameterValue(
                    LaunchConfiguration("fp_trigger_track_frames"),
                    value_type=int,
                ),

                "fp_trigger_track_use_new_frames": ParameterValue(
                    LaunchConfiguration("fp_trigger_track_use_new_frames"),
                    value_type=bool,
                ),

                "fp_debug": ParameterValue(
                    LaunchConfiguration("fp_debug"),
                    value_type=int,
                ),

                "fp_debug_dir": LaunchConfiguration("fp_debug_dir"),

                # ----------------------------
                # Empty-space
                # ----------------------------
                "empty_space_topic": LaunchConfiguration("empty_space_topic"),

                "empty_space_enable": ParameterValue(
                    LaunchConfiguration("empty_space_enable"),
                    value_type=bool,
                ),

                "empty_grid_step_px": ParameterValue(
                    LaunchConfiguration("empty_grid_step_px"),
                    value_type=int,
                ),

                "empty_max_candidates": ParameterValue(
                    LaunchConfiguration("empty_max_candidates"),
                    value_type=int,
                ),

                "empty_roi_x_min": ParameterValue(
                    LaunchConfiguration("empty_roi_x_min"),
                    value_type=int,
                ),

                "empty_roi_y_min": ParameterValue(
                    LaunchConfiguration("empty_roi_y_min"),
                    value_type=int,
                ),

                "empty_roi_x_max": ParameterValue(
                    LaunchConfiguration("empty_roi_x_max"),
                    value_type=int,
                ),

                "empty_roi_y_max": ParameterValue(
                    LaunchConfiguration("empty_roi_y_max"),
                    value_type=int,
                ),

                "empty_roi_right_margin": ParameterValue(
                    LaunchConfiguration("empty_roi_right_margin"),
                    value_type=int,
                ),

                "empty_roi_bottom_margin": ParameterValue(
                    LaunchConfiguration("empty_roi_bottom_margin"),
                    value_type=int,
                ),

                "empty_mask_dilate_px": ParameterValue(
                    LaunchConfiguration("empty_mask_dilate_px"),
                    value_type=int,
                ),

                "empty_depth_patch_radius_px": ParameterValue(
                    LaunchConfiguration("empty_depth_patch_radius_px"),
                    value_type=int,
                ),

                "empty_depth_valid_ratio_min": ParameterValue(
                    LaunchConfiguration("empty_depth_valid_ratio_min"),
                    value_type=float,
                ),

                "empty_depth_spread_max_m": ParameterValue(
                    LaunchConfiguration("empty_depth_spread_max_m"),
                    value_type=float,
                ),

                "empty_space_vis_hold_sec": ParameterValue(
                    LaunchConfiguration("empty_space_vis_hold_sec"),
                    value_type=float,
                ),

                # ----------------------------
                # Priority debug
                # ----------------------------
                "priority_debug_save": ParameterValue(
                    LaunchConfiguration("priority_debug_save"),
                    value_type=bool,
                ),

                "priority_debug_dir": LaunchConfiguration("priority_debug_dir"),

                # ----------------------------
                # Topics
                # ----------------------------
                "object_topic": LaunchConfiguration("object_topic"),
                "insert_topic": LaunchConfiguration("insert_topic"),

                "object_pose_topic": LaunchConfiguration("object_pose_topic"),
                "insert_pose_topic": LaunchConfiguration("insert_pose_topic"),

                "detect_mode_topic": LaunchConfiguration("detect_mode_topic"),
                "object_trigger_topic": LaunchConfiguration("object_trigger_topic"),
            }],
        ),
    ])