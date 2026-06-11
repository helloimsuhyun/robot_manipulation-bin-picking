from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 여기 있는 값들을 수정하면 main.py를 직접 수정하지 않고 동작 파라미터를 바꿀 수 있습니다.
    peg_in_hole_params = {
        # ===== 로봇/ROS 설정 =====
        'robot_ip': '192.168.1.10',             # 연구실: 169.254.186.20 / 111호: 192.168.1.10
        'use_simulation_mode': False,           # 실제 로봇 구동 시 False로 변경 검토
        'gripper_topic': '/grip_state',         # 절대 토픽 권장

        # grip_current.py 주석 기준: 1=open, 0=close, 2=stop
        'grip_open': 1,
        'grip_close': 0,
        'grip_stop': 2,

        # ===== 고정 joint 자세 =====
        'home_joint': [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0],
        'peg_camera_joint': [10.87, 2.78, 79.15, 8.07, 90.0, 34.16],
        'hole_camera_joint': [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16],
        'peg_return_mid_joint': [-47.0, 2.78, 79.15, 8.07, 90.0, 34.16],

        # ===== 작업 z 좌표 =====
        'pick_down_target_z_mm': 69.83,
        'pick_approach_offset_z_mm': 30.0,
        'pick_up_target_z_mm': 110.0,

        'place_approach_target_z_mm': 108.0,
        'place_down_target_z_mm': 98.0,
        'place_up_target_z_mm': 110.0,

        # ===== 속도/가속도 =====
        'move_j_speed': 30.0,
        'move_j_acc': 40.0,
        'move_l_speed': 40.0,
        'move_l_acc': 60.0,

        # ===== 비전 토픽 =====
        # trigger_*: 컨트롤러가 현재 TCP pose [x,y,z,rx,ry,rz]를 publish
        # *_targets: 비전 노드가 [x,y,yaw,id, ...]를 publish
        'peg_targets_topic': '/vision/peg_targets',
        'hole_targets_topic': '/vision/hole_targets',
        'trigger_peg_topic': '/manipulation/trigger_peg',
        'trigger_hole_topic': '/manipulation/trigger_hole',
        'camera_settle_sec': 0.5,
        'vision_wait_timeout_sec': 6.0,

        # ===== 대기/timeout =====
        'grasp_wait_sec': 1.0,
        'release_wait_sec': 1.0,
        'move_start_timeout_sec': 3.0,
    }

    return LaunchDescription([
        Node(
            package='control',
            executable='peg_in_hole_controller',
            name='peg_in_hole_controller',
            output='screen',
            parameters=[peg_in_hole_params],
        )
    ])
