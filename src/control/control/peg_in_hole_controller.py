import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

try:
    from .task_types import TaskContext, TaskState, VisionTarget
    from .robot_motion import RobotMotion
    from .gripper_interface import GripperInterface
    from .vision_interface import VisionInterface
except ImportError:  # direct script/debug execution support
    from task_types import TaskContext, TaskState, VisionTarget
    from robot_motion import RobotMotion
    from gripper_interface import GripperInterface
    from vision_interface import VisionInterface


class PegInHoleController(Node):
    """
    상태머신만 담당하는 peg-in-hole 메인 컨트롤러.

    - 로봇 이동: RobotMotion
    - 그리퍼 publish: GripperInterface
    - 비전 trigger/subscribe/parsing: VisionInterface

    기본 trigger 토픽 데이터:
        Float64MultiArray.data = [x, y, z, rx, ry, rz]

    6D peg trigger 토픽 데이터:
        Float64MultiArray.data = [object_id, x, y, z, rx, ry, rz]

    기본 vision 결과 토픽 데이터:
        Float64MultiArray.data = [x, y, yaw, id, x, y, yaw, id, ...]
        id: 0=cylinder, 1=hole, 2=cross

    6D peg 결과 토픽 데이터:
        success: len(data) == 6  -> [x, y, z, rx, ry, rz]
            - 비전 노드에서 이미 move_l용 최종 grasp pose를 계산해서 보낸다.
        legacy:  len(data) == 17 -> [4x4 target transform row-major, object_id]
        failure: 그 외 길이 -> [currently visible object ids]
    """

    def __init__(self):
        super().__init__("peg_in_hole_controller")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("robot_ip", "192.168.1.10"),
                ("use_simulation_mode", False),
                ("gripper_topic", "/grip_state"),

                ("grip_open", 1),
                ("grip_close", 0),
                ("grip_stop", 2),

                ("home_joint", [-90.0, 0.0, 90.0, 0.0, 90.0, 45.0]),
                ("peg_camera_joint", [10.87, 2.78, 79.15, 8.07, 90.0, 34.16]),
                ("hole_camera_joint", [-169.23, 2.78, 79.15, 8.07, 90.0, 34.16]),

                # MoveL에서 항상 강제할 수평 TCP 자세
                ("flat_tcp_rx_deg", 90.0),
                ("flat_tcp_ry_deg", 0.0),
                ("flat_tcp_rz_deg", 0.0),

                ("pick_down_target_z_mm", 69.83),
                ("pick_approach_offset_z_mm", 30.0),
                ("pick_up_target_z_mm", 110.0),

                # 6D peg 방식: 비전에서 받은 peg/grasp pose의 local z축 기준 offset
                ("pick_approach_above_peg_z_mm", 160.0),
                ("pick_grasp_above_peg_z_mm", 150.0),
                ("pick_lift_above_peg_z_mm", 180.0),

                ("place_approach_target_z_mm", 108.0),
                ("place_down_target_z_mm", 98.0),
                ("place_up_target_z_mm", 110.0),

                # place tilt / servo_t sequence
                ("place_tilt_deg", 7.0),
                ("place_lift_current_tcp_z_mm", 30.0),

                ("servo_t_t1_sec", 0.01),
                ("servo_t_t2_sec", 0.05),
                ("servo_t_sleep_sec", 0.01),
                ("servo_t_compensation", 1),

                # 약한 삽입 힘은 현재 joint torque vector로 지정한다.
                # Cartesian -Z force가 아니라 servo_t 외부 joint torque이므로 반드시 작게 시작한다.
                ("servo_insert_duration_sec", 2.0),
                ("servo_insert_down_torque", [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),

                # 4/5번 조인트 자세 복귀 servo_t
                ("servo_level_max_duration_sec", 3.0),
                ("servo_level_target_stable_count", 10),
                ("servo_level_j4_tol_deg", 1.0),
                ("servo_level_j5_tol_deg", 1.0),
                ("servo_level_q5_des_deg", 90.0),
                ("servo_level_kp_j4", 1.5),
                ("servo_level_kd_j4", 0.25),
                ("servo_level_kp_j5", 1.1),
                ("servo_level_kd_j5", 0.16),
                ("servo_level_max_j4_torque", 15.0),
                ("servo_level_max_j5_torque", 12.0),
                ("servo_level_j4_sign", 1.0),
                ("servo_level_j5_sign", 1.0),
                ("servo_level_j4_deadband_deg", 0.3),
                ("servo_level_j5_deadband_deg", 0.3),
                ("servo_jvel_lpf_alpha", 0.1),
                ("servo_torque_rate_limit", 0.16),
                ("servo_torque_lpf_alpha", 0.30),

                ("move_j_speed", 60.0),
                ("move_j_acc", 80.0),
                ("move_l_speed", 80.0),
                ("move_l_acc", 120.0),

                # peg/hole 공통 접근 MoveL 속도
                ("approach_move_l_speed", 40.0),
                ("approach_move_l_acc", 80.0),

                # peg/hole 공통 하강 MoveL 속도
                ("descend_move_l_speed", 8.0),
                ("descend_move_l_acc", 40.0),

                ("move_start_timeout_sec", 0.5),

                ("joint_tol_deg", 0.5),
                ("joint_stable_count_required", 5),
                ("joint_wait_timeout_sec", 40.0),
                ("joint_polling_dt_sec", 0.05),

                ("tcp_pos_tol_mm", 1.0),
                ("tcp_rot_tol_deg", 2.0),
                ("tcp_stable_count_required", 5),
                ("tcp_wait_timeout_sec", 40.0),
                ("tcp_polling_dt_sec", 0.05),

                ("grasp_wait_sec", 1.0),
                ("release_wait_sec", 1.0),

                ("peg_targets_topic", "/vision/peg_targets"),
                ("hole_targets_topic", "/vision/hole_targets"),
                ("trigger_peg_topic", "/manipulation/trigger_peg"),
                ("trigger_hole_topic", "/manipulation/trigger_hole"),
                ("camera_settle_sec", 0.5),
                ("use_6d_peg_interface", False),
                ("pause_before_peg_inspect", True),
                ("manual_continue_topic", "/manual_continue"),

                ("vision_wait_timeout_sec", 2.0),
                ("vision_fixed_rx_deg", 90.0),
                ("vision_fixed_rz_deg", 0.0),

                # 방금 원위치에 되돌려놓은 peg를 한 번 제외할 때 쓰는 xy 거리 기준
                ("skip_once_xy_tol_mm", 20.0),
                
            ],
        )

        self.state = TaskState.IDLE_HOME
        self.use_simulation_mode = self._get_bool_param("use_simulation_mode")

        robot_ip = self._get_str_param("robot_ip")
        gripper_topic = self._get_str_param("gripper_topic")
        peg_targets_topic = self._get_str_param("peg_targets_topic")
        hole_targets_topic = self._get_str_param("hole_targets_topic")
        trigger_peg_topic = self._get_str_param("trigger_peg_topic")
        trigger_hole_topic = self._get_str_param("trigger_hole_topic")
        camera_settle_sec = self._get_float_param("camera_settle_sec")
        use_6d_peg_interface = self._get_bool_param("use_6d_peg_interface")
        self.use_6d_peg_interface = use_6d_peg_interface
        manual_continue_topic = self._get_str_param("manual_continue_topic")

        self.ctx = TaskContext(
            home_joint=self._get_array_param("home_joint", 6),
            peg_camera_joint=self._get_array_param("peg_camera_joint", 6),
            hole_camera_joint=self._get_array_param("hole_camera_joint", 6),

            flat_tcp_rx_deg=self._get_float_param("flat_tcp_rx_deg"),
            flat_tcp_ry_deg=self._get_float_param("flat_tcp_ry_deg"),
            flat_tcp_rz_deg=self._get_float_param("flat_tcp_rz_deg"),

            pick_down_target_z_mm=self._get_float_param("pick_down_target_z_mm"),
            pick_approach_offset_z_mm=self._get_float_param("pick_approach_offset_z_mm"),
            pick_up_target_z_mm=self._get_float_param("pick_up_target_z_mm"),

            pick_approach_above_peg_z_mm=self._get_float_param("pick_approach_above_peg_z_mm"),
            pick_grasp_above_peg_z_mm=self._get_float_param("pick_grasp_above_peg_z_mm"),
            pick_lift_above_peg_z_mm=self._get_float_param("pick_lift_above_peg_z_mm"),

            place_approach_target_z_mm=self._get_float_param("place_approach_target_z_mm"),
            place_down_target_z_mm=self._get_float_param("place_down_target_z_mm"),
            place_up_target_z_mm=self._get_float_param("place_up_target_z_mm"),

            place_tilt_deg=self._get_float_param("place_tilt_deg"),
            place_lift_current_tcp_z_mm=self._get_float_param("place_lift_current_tcp_z_mm"),

            servo_t_t1_sec=self._get_float_param("servo_t_t1_sec"),
            servo_t_t2_sec=self._get_float_param("servo_t_t2_sec"),
            servo_t_sleep_sec=self._get_float_param("servo_t_sleep_sec"),
            servo_t_compensation=self._get_int_param("servo_t_compensation"),

            servo_insert_duration_sec=self._get_float_param("servo_insert_duration_sec"),
            servo_insert_down_torque=self._get_array_param("servo_insert_down_torque", 6),

            servo_level_max_duration_sec=self._get_float_param("servo_level_max_duration_sec"),
            servo_level_target_stable_count=self._get_int_param("servo_level_target_stable_count"),
            servo_level_j4_tol_deg=self._get_float_param("servo_level_j4_tol_deg"),
            servo_level_j5_tol_deg=self._get_float_param("servo_level_j5_tol_deg"),
            servo_level_q5_des_deg=self._get_float_param("servo_level_q5_des_deg"),
            servo_level_kp_j4=self._get_float_param("servo_level_kp_j4"),
            servo_level_kd_j4=self._get_float_param("servo_level_kd_j4"),
            servo_level_kp_j5=self._get_float_param("servo_level_kp_j5"),
            servo_level_kd_j5=self._get_float_param("servo_level_kd_j5"),
            servo_level_max_j4_torque=self._get_float_param("servo_level_max_j4_torque"),
            servo_level_max_j5_torque=self._get_float_param("servo_level_max_j5_torque"),
            servo_level_j4_sign=self._get_float_param("servo_level_j4_sign"),
            servo_level_j5_sign=self._get_float_param("servo_level_j5_sign"),
            servo_level_j4_deadband_deg=self._get_float_param("servo_level_j4_deadband_deg"),
            servo_level_j5_deadband_deg=self._get_float_param("servo_level_j5_deadband_deg"),
            servo_jvel_lpf_alpha=self._get_float_param("servo_jvel_lpf_alpha"),
            servo_torque_rate_limit=self._get_float_param("servo_torque_rate_limit"),
            servo_torque_lpf_alpha=self._get_float_param("servo_torque_lpf_alpha"),

            move_j_speed=self._get_float_param("move_j_speed"),
            move_j_acc=self._get_float_param("move_j_acc"),
            move_l_speed=self._get_float_param("move_l_speed"),
            move_l_acc=self._get_float_param("move_l_acc"),

            approach_move_l_speed=self._get_float_param("approach_move_l_speed"),
            approach_move_l_acc=self._get_float_param("approach_move_l_acc"),
            descend_move_l_speed=self._get_float_param("descend_move_l_speed"),
            descend_move_l_acc=self._get_float_param("descend_move_l_acc"),

            move_start_timeout_sec=self._get_float_param("move_start_timeout_sec"),

            joint_tol_deg=self._get_float_param("joint_tol_deg"),
            joint_stable_count_required=self._get_int_param("joint_stable_count_required"),
            joint_wait_timeout_sec=self._get_float_param("joint_wait_timeout_sec"),
            joint_polling_dt_sec=self._get_float_param("joint_polling_dt_sec"),

            tcp_pos_tol_mm=self._get_float_param("tcp_pos_tol_mm"),
            tcp_rot_tol_deg=self._get_float_param("tcp_rot_tol_deg"),
            tcp_stable_count_required=self._get_int_param("tcp_stable_count_required"),
            tcp_wait_timeout_sec=self._get_float_param("tcp_wait_timeout_sec"),
            tcp_polling_dt_sec=self._get_float_param("tcp_polling_dt_sec"),

            grasp_wait_sec=self._get_float_param("grasp_wait_sec"),
            release_wait_sec=self._get_float_param("release_wait_sec"),

            vision_wait_timeout_sec=self._get_float_param("vision_wait_timeout_sec"),
            vision_fixed_rx_deg=self._get_float_param("vision_fixed_rx_deg"),
            vision_fixed_rz_deg=self._get_float_param("vision_fixed_rz_deg"),

            skip_once_xy_tol_mm=self._get_float_param("skip_once_xy_tol_mm"),
        )

        self.manual_continue_received = False
        self.pause_before_next_peg_inspect = False

        self.manual_continue_sub = self.create_subscription(
            Empty,
            manual_continue_topic,
            self.manual_continue_callback,
            10,
        )

        self.motion = RobotMotion(
            node=self,
            ctx=self.ctx,
            robot_ip=robot_ip,
            use_simulation_mode=self.use_simulation_mode,
        )

        self.gripper = GripperInterface(
            node=self,
            topic=gripper_topic,
            grip_open=self._get_int_param("grip_open"),
            grip_close=self._get_int_param("grip_close"),
            grip_stop=self._get_int_param("grip_stop"),
        )

        self.vision = VisionInterface(
            node=self,
            ctx=self.ctx,
            robot_motion=self.motion,
            peg_targets_topic=peg_targets_topic,
            hole_targets_topic=hole_targets_topic,
            trigger_peg_topic=trigger_peg_topic,
            trigger_hole_topic=trigger_hole_topic,
            camera_settle_sec=camera_settle_sec,
            use_6d_peg_interface=use_6d_peg_interface,
        )

        self.get_logger().info(f"Robot IP: {robot_ip}")
        self.get_logger().info(f"Gripper topic: {gripper_topic}")
        self.get_logger().info(f"Peg target topic: {peg_targets_topic}")
        self.get_logger().info(f"Hole target topic: {hole_targets_topic}")
        self.get_logger().info(f"Trigger peg topic: {trigger_peg_topic}")
        self.get_logger().info(f"Trigger hole topic: {trigger_hole_topic}")
        self.get_logger().info(f"Manual continue topic: {manual_continue_topic}")
        self.get_logger().info(f"Camera settle sec: {camera_settle_sec}")
        self.get_logger().info(f"Use 6D peg interface: {use_6d_peg_interface}")
        self.get_logger().info(f"Use simulation mode: {self.use_simulation_mode}")
        self.get_logger().info(
            f"Flat TCP RPY: "
            f"[{self.ctx.flat_tcp_rx_deg}, "
            f"{self.ctx.flat_tcp_ry_deg}, "
            f"{self.ctx.flat_tcp_rz_deg}]"
        )
        self.get_logger().info(
            f"Approach MoveL speed/acc: "
            f"{self.ctx.approach_move_l_speed}, "
            f"{self.ctx.approach_move_l_acc}"
        )
        self.get_logger().info(
            f"Descend MoveL speed/acc: "
            f"{self.ctx.descend_move_l_speed}, "
            f"{self.ctx.descend_move_l_acc}"
        )
        self.get_logger().info(
            f"Place tilt/servo: tilt={self.ctx.place_tilt_deg} deg, "
            f"insert_duration={self.ctx.servo_insert_duration_sec} sec, "
            f"insert_torque={np.round(self.ctx.servo_insert_down_torque, 4).tolist()}, "
            f"level_tol=[{self.ctx.servo_level_j4_tol_deg}, {self.ctx.servo_level_j5_tol_deg}] deg"
        )

    # ------------------------------------------------------------------
    # parameter helper
    # ------------------------------------------------------------------
    def _get_str_param(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _get_bool_param(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _get_int_param(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _get_float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _get_array_param(self, name: str, expected_len: int) -> np.ndarray:
        value = list(self.get_parameter(name).value)

        if len(value) != expected_len:
            raise ValueError(
                f"Parameter '{name}' must have length {expected_len}, but got {len(value)}"
            )

        return np.array(value, dtype=float)

    # ------------------------------------------------------------------
    # manual continue helper
    # ------------------------------------------------------------------
    def manual_continue_callback(self, msg):
        self.manual_continue_received = True
        self.get_logger().info("[PAUSE] Manual continue signal received.")

    def wait_for_space_before_peg_inspect(self):
        pause_enabled = self._get_bool_param("pause_before_peg_inspect")

        self.get_logger().info(
            f"[PAUSE DEBUG] pause_before_peg_inspect = {pause_enabled}, "
            f"pause_before_next_peg_inspect = {self.pause_before_next_peg_inspect}, "
            f"active_jig_targets = {len(self.ctx.active_jig_targets)}"
        )

        if not pause_enabled:
            self.get_logger().info(
                "[PAUSE] Skip pause because pause_before_peg_inspect is False."
            )
            return

        if not self.pause_before_next_peg_inspect:
            self.get_logger().info(
                "[PAUSE] Skip pause because previous jig layout is not fully filled."
            )
            return

        self.manual_continue_received = False

        self.get_logger().info(
            "[PAUSE] Previous jig layout is fully filled. "
            "Waiting for /manual_continue before inspecting pegs."
        )

        while rclpy.ok() and not self.manual_continue_received:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.pause_before_next_peg_inspect = False

        self.get_logger().info("[PAUSE] Continue. Start peg inspection.")

    # ------------------------------------------------------------------
    # selection helper
    # ------------------------------------------------------------------
    def is_skip_once_target(self, peg) -> bool:
        """
        matching jig가 없어서 원래 위치에 다시 내려놓은 peg와
        같은 id의 peg는 다음 선택에서 딱 1번만 제외한다.

        위치 기준이 아니라 id 기준이다.
        예:
            방금 원 peg를 되돌려놓음
            → 다음 선택에서 object_id == 0인 peg는 모두 한 번 제외
        """
        if self.ctx.skip_once_pick_id is None:
            return False

        if peg.object_id == self.ctx.skip_once_pick_id:
            self.get_logger().info(
                f"[SKIP_ONCE] skip same id once. "
                f"id = {peg.object_id} "
                f"({self.vision.shape_name(peg.object_id)})"
            )
            return True

        return False

    def _set_current_peg(self, peg_index: int):
        selected_peg = self.ctx.peg_targets[peg_index]

        self.ctx.current_peg_index = peg_index
        self.ctx.current_peg_pick_pose = selected_peg.pose.copy()
        self.ctx.current_peg_object_T = (
            selected_peg.transform.copy()
            if selected_peg.transform is not None
            else None
        )
        self.ctx.current_target_id = selected_peg.object_id

        self.get_logger().info(
            f"[SELECT] current peg index = {self.ctx.current_peg_index}, "
            f"id = {self.ctx.current_target_id} "
            f"({self.vision.shape_name(self.ctx.current_target_id)}), "
            f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
        )


    def get_preferred_peg_request_ids(self) -> list[int]:
        """
        6D peg 비전에는 "잡고 싶은 id"가 아니라 "잡아도 되는 id 목록"을 보낸다.

        publish 쪽에서는 이 목록을 아래 mask로 변환한다.
            [allow_circle, allow_square, allow_cross, pose6]

        - 아직 jig layout을 모르면 모든 peg를 잡아도 되므로 [0, 1, 2]를 허용한다.
        - jig layout을 이미 알고 있으면 남은 jig id만 허용한다.
        - skip_once id는 가능한 한 이번 허용 후보에서 제외한다.
        """
        if len(self.ctx.active_jig_targets) == 0:
            request_ids = [0, 1, 2]
        else:
            request_ids = [
                int(object_id)
                for object_id, count in self.ctx.remaining_jig_counts.items()
                if count > 0
            ]

        if self.ctx.skip_once_pick_id is not None and len(request_ids) > 1:
            request_ids = [
                object_id
                for object_id in request_ids
                if object_id != self.ctx.skip_once_pick_id
            ]

        if len(request_ids) == 0:
            request_ids = [0, 1, 2]

        return request_ids

    def select_next_peg(self) -> bool:
        if len(self.ctx.peg_targets) == 0:
            self.ctx.current_peg_index = -1
            self.ctx.current_peg_pick_pose = None
            self.ctx.current_peg_object_T = None
            self.ctx.current_target_id = None
            return False

        selected_index = None

        # active_jig_targets가 비어 있으면 아직 현재 jig 세트를 모르는 상태다.
        # 이때는 기존처럼 첫 번째 peg를 잡고, 이후 hole 촬영으로 jig 세트를 확정한다.
        if len(self.ctx.active_jig_targets) == 0:
            self.get_logger().info(
                "[SELECT] No active jig layout. Select first non-skipped peg."
            )

            for i, peg in enumerate(self.ctx.peg_targets):
                if self.is_skip_once_target(peg):
                    continue

                selected_index = i
                break

            if selected_index is None:
                self.get_logger().warn(
                    "[SELECT] Only skipped peg is available. "
                    "Clear skip_once and select first peg."
                )
                self.clear_skip_once()
                selected_index = 0

            self._set_current_peg(selected_index)
            self.clear_skip_once()
            return True

        # active_jig_targets가 남아 있으면, 해당 jig 세트가 다 찰 때까지
        # 저장된 남은 jig 타입에 맞는 peg만 선택한다.
        for i, peg in enumerate(self.ctx.peg_targets):
            if self.ctx.remaining_jig_counts.get(peg.object_id, 0) <= 0:
                continue

            if self.is_skip_once_target(peg):
                continue

            selected_index = i
            break

        if selected_index is not None:
            peg = self.ctx.peg_targets[selected_index]
            self.get_logger().info(
                f"[SELECT] Select peg matched with active jig id = {peg.object_id} "
                f"({self.vision.shape_name(peg.object_id)}). "
                f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
            )
            self._set_current_peg(selected_index)
            self.clear_skip_once()
            return True

        # skip_once 때문에 선택할 peg가 없을 수도 있다.
        # 이 경우 한 번 제외 조건을 해제하고 다시 남은 jig 타입 기준으로 선택한다.
        if self.ctx.skip_once_pick_id is not None:
            self.get_logger().warn(
                "[SELECT] No candidate after skip_once. "
                "Clear skip_once and retry selection."
            )
            self.clear_skip_once()

            for i, peg in enumerate(self.ctx.peg_targets):
                if self.ctx.remaining_jig_counts.get(peg.object_id, 0) > 0:
                    selected_index = i
                    break

            if selected_index is not None:
                self._set_current_peg(selected_index)
                return True

        # 여기까지 왔다는 것은 현재 저장된 jig 세트에 맞는 peg가 보이지 않는다는 의미다.
        # 기존처럼 jig 정보를 지우고 아무 peg나 잡으면 같은 jig 세트를 끝까지 채우지 못하므로 멈춘다.
        self.ctx.current_peg_index = -1
        self.ctx.current_peg_pick_pose = None
        self.ctx.current_peg_object_T = None
        self.ctx.current_target_id = None

        self.get_logger().warn(
            "[SELECT] No peg matched with active jig layout. "
            f"Need jig ids = {dict(self.ctx.remaining_jig_counts)}. "
            "Stop current task instead of clearing jig memory."
        )
        return False

    def _initialize_active_jig_layout_if_needed(self):
        """
        active_jig_targets가 비어 있을 때만 현재 촬영된 hole 목록을 저장한다.

        의미:
            - 비어 있음: 새 jig 세트 시작
            - 비어 있지 않음: 이전에 본 jig 세트가 아직 덜 찼으므로 위치/개수를 유지

        예:
            처음 hole 촬영 결과가 [네모, 동그라미, 네모, 십자가]이면
            active_jig_targets에 4개 slot 위치를 모두 저장한다.
            이후 4개가 모두 찰 때까지 이 목록에서 하나씩 제거하며 사용한다.
        """
        if len(self.ctx.active_jig_targets) != 0:
            self.get_logger().info(
                "[JIG] Keep active jig layout. "
                f"remaining slot count = {len(self.ctx.active_jig_targets)}, "
                f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
            )
            return

        self.ctx.active_jig_targets = [
            VisionTarget(
                pose=target.pose.copy(),
                object_id=target.object_id,
            )
            for target in self.ctx.hole_targets
        ]

        self.ctx.remaining_jig_counts = Counter(
            target.object_id for target in self.ctx.active_jig_targets
        )

        self.get_logger().info(
            f"[JIG] Initialize active jig layout. "
            f"slot_count = {len(self.ctx.active_jig_targets)}, "
            f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
        )

        for i, target in enumerate(self.ctx.active_jig_targets):
            self.get_logger().info(
                f"[JIG] slot[{i}] id = {target.object_id} "
                f"({self.vision.shape_name(target.object_id)}), "
                f"pose = {target.pose}"
            )

    def select_next_hole(self) -> bool:
        if self.ctx.current_target_id is None:
            raise RuntimeError("No selected peg id. Cannot select matching hole.")

        if len(self.ctx.active_jig_targets) == 0:
            self.ctx.current_hole_index = -1
            self.ctx.current_hole_place_pose = None
            self.get_logger().warn("[SELECT] No available active jig slot")
            return False

        self.ctx.current_hole_index = -1
        self.ctx.current_hole_place_pose = None

        for i, hole in enumerate(self.ctx.active_jig_targets):
            if hole.object_id == self.ctx.current_target_id:
                self.ctx.current_hole_index = i
                self.ctx.current_hole_place_pose = hole.pose.copy()

                self.get_logger().info(
                    f"[SELECT] matched active jig slot index = {i}, "
                    f"id = {self.ctx.current_target_id} "
                    f"({self.vision.shape_name(self.ctx.current_target_id)}), "
                    f"pose = {self.ctx.current_hole_place_pose}"
                )
                return True

        available_ids = [target.object_id for target in self.ctx.active_jig_targets]

        self.get_logger().warn(
            f"[SELECT] No matching hole found for selected peg id = "
            f"{self.ctx.current_target_id} "
            f"({self.vision.shape_name(self.ctx.current_target_id)}). "
            f"Available hole ids = {available_ids}"
        )

        if self.ctx.current_target_id in self.ctx.remaining_jig_counts:
            del self.ctx.remaining_jig_counts[self.ctx.current_target_id]

            self.get_logger().warn(
                f"[JIG] Remove unavailable jig id = {self.ctx.current_target_id} "
                f"from remaining_jig_counts. "
                f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
            )

        return False

    def mark_current_jig_used(self):
        if self.ctx.current_target_id is None:
            return

        object_id = self.ctx.current_target_id

        if self.ctx.remaining_jig_counts.get(object_id, 0) > 0:
            self.ctx.remaining_jig_counts[object_id] -= 1

            if self.ctx.remaining_jig_counts[object_id] <= 0:
                del self.ctx.remaining_jig_counts[object_id]

        self.get_logger().info(
            f"[JIG] Used jig id = {object_id} "
            f"({self.vision.shape_name(object_id)}), "
            f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
        )

    @staticmethod
    def _normalize_vec(v: np.ndarray, name: str) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            raise ValueError(f"{name} norm is too small: {n}")
        return v / n

    @staticmethod
    def _orthonormalize_R(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float).reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0.0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    def _R_to_euler_zyx_deg(self, R: np.ndarray) -> np.ndarray:
        """
        R = Rz(rz) @ Ry(ry) @ Rx(rx) 기준으로 [rx, ry, rz] deg를 복원한다.
        vision_interface.py 및 sixd_pose_transform_node.py와 같은 규칙이다.
        """
        R = self._orthonormalize_R(R)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy >= 1e-9:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0

        return np.degrees([rx, ry, rz]).astype(float)
    
    def _rot_x_deg(self, deg: float) -> np.ndarray:
        a = np.deg2rad(float(deg))
        c = np.cos(a)
        s = np.sin(a)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=float,
        )

    def _normalize_yaw_deg(self, yaw: float) -> float:
        """
        yaw를 -180 ~ 180 deg 범위로 정규화한다.
        """
        return (float(yaw) + 180.0) % 360.0 - 180.0

    def _normalize_yaw_0_to_90_deg(self, yaw: float) -> float:
        """
        90도 대칭인 네모/십자가 place yaw를 0 <= rz < 90 deg 범위로 접는다.
        예:
            -10 -> 80
             95 -> 5
            180 -> 0
        """
        return float(yaw) % 90.0

    def _correct_pick_yaw_6d(self, yaw: float, object_id: int) -> float:
        """
        6D peg pick에서 object frame -> TCP pose 변환 후 나온 rz(yaw)를
        object_id 기준으로 후처리한다.

        id 의미:
            0: 원
            1: 사각형
            2: 십자가
        """
        yaw = float(yaw)
        object_id = int(object_id)

        if object_id == 0:
            corrected_yaw = 135.0

        elif object_id == 1:
            corrected_yaw = (yaw % 90.0) + 135.0

        elif object_id == 2:
            corrected_yaw = (yaw % 90.0) + 135.0

        else:
            self.get_logger().warn(
                f"[6D PICK YAW] Unknown object_id={object_id}. "
                f"Use raw yaw={yaw:.3f}"
            )
            corrected_yaw = yaw

        corrected_yaw = self._normalize_yaw_deg(corrected_yaw)

        self.get_logger().info(
            f"[6D PICK YAW] id={object_id}, "
            f"raw_yaw={yaw:.3f}, "
            f"corrected_yaw={corrected_yaw:.3f}"
        )

        return corrected_yaw

    def _make_tcp_pick_pose_from_object_T(
        self,
        base_T_object: np.ndarray,
        offset_mm: float,
        object_id: int,
    ) -> np.ndarray:
        """
        6D vision에서 받은 base_T_object/grasp를 기준으로 pick용 TCP pose를 만든다.

        의도:
            - object local +Z 방향으로 offset 위치를 만든다.
              따라서 offset 160 -> 150으로 이동하면 실제 접근 방향은 object local -Z.
            - object local X축 방향을 그리퍼 정렬 방향으로 사용한다.
            - rbpodo move_l TCP convention에 맞추기 위해 Rx(90deg) 보정을 적용한다.
            - 마지막으로 object_id 기준 yaw 후처리를 적용한다.

        반환:
            pose6 = [x, y, z, rx, ry, rz]
        """
        base_T_object = np.asarray(base_T_object, dtype=np.float64).reshape(4, 4)

        R_obj = base_T_object[:3, :3]
        p_obj = base_T_object[:3, 3]

        R_obj = self._orthonormalize_R(R_obj)

        object_x = R_obj[:, 0]
        object_z = R_obj[:, 2]

        # object X축을 gripper 정렬축으로 사용
        tcp_x = object_x / np.linalg.norm(object_x)

        # object Z축을 접근 기준축으로 사용
        tcp_z = object_z / np.linalg.norm(object_z)

        # 수치 오차 때문에 x/z가 완전히 직교하지 않을 수 있으므로 재직교화
        tcp_y = np.cross(tcp_z, tcp_x)
        tcp_y_norm = np.linalg.norm(tcp_y)

        if tcp_y_norm < 1e-9:
            self.get_logger().warn(
                "[6D PICK POSE] object_x and object_z are nearly parallel. "
                "Use object rotation directly."
            )
            R_aligned = R_obj.copy()
            tcp_z = R_aligned[:, 2]
        else:
            tcp_y = tcp_y / tcp_y_norm
            tcp_x = np.cross(tcp_y, tcp_z)
            tcp_x = tcp_x / np.linalg.norm(tcp_x)

            R_aligned = np.column_stack((tcp_x, tcp_y, tcp_z))

        R_aligned = self._orthonormalize_R(R_aligned)

        # rbpodo move_l TCP RPY convention 보정
        # 평평한 top-down 자세가 rx ~= 90deg 계열이므로,
        # object/grasp frame을 그대로 Euler로 바꾸지 않고 local X축 +90deg 보정.
        R_tcp = R_aligned @ self._rot_x_deg(90.0)
        R_tcp = self._orthonormalize_R(R_tcp)

        # object/grasp local +Z 방향으로 offset을 둔다.
        # offset 160 -> 150으로 이동하면 실제 이동 방향은 local -Z.
        p_tcp = p_obj + tcp_z * float(offset_mm)

        rpy = self._R_to_euler_zyx_deg(R_tcp)

        raw_yaw = float(rpy[2])
        rpy[2] = self._correct_pick_yaw_6d(
            yaw=raw_yaw,
            object_id=int(object_id),
        )

        pose6 = np.array(
            [
                float(p_tcp[0]),
                float(p_tcp[1]),
                float(p_tcp[2]),
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]),
            ],
            dtype=float,
        )

        self.get_logger().info(
            f"[6D PICK POSE] offset={float(offset_mm):.1f}mm, "
            f"id={int(object_id)}, "
            f"object_p={np.round(p_obj, 3)}, "
            f"object_x={np.round(object_x, 4)}, "
            f"object_z={np.round(object_z, 4)}, "
            f"raw_yaw={raw_yaw:.3f}, "
            f"corrected_yaw={rpy[2]:.3f}, "
            f"tcp_pose={np.round(pose6, 3)}"
        )

        return pose6


    def _pose6_to_R_zyx(self, pose6: np.ndarray) -> np.ndarray:
        """
        pose6 = [x, y, z, rx, ry, rz]를 회전행렬로 변환한다.

        규칙:
            R = Rz(rz) @ Ry(ry) @ Rx(rx)

        주의:
            이 함수는 비전에서 받은 grasp pose의 local Z축을 구하기 위해서만 쓴다.
            rx, ry, rz 자체는 제어부에서 보정하지 않고 그대로 move_l에 넘긴다.
        """
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        rx, ry, rz = np.deg2rad(pose6[3:6])

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx, cx],
            ],
            dtype=float,
        )

        Ry = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ],
            dtype=float,
        )

        Rz = np.array(
            [
                [cz, -sz, 0.0],
                [sz, cz, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        return Rz @ Ry @ Rx

    def _make_pose_offset_along_local_y(
        self,
        grasp_pose: np.ndarray,
        offset_mm: float,
    ) -> np.ndarray:
        """
        비전에서 받은 최종 grasp pose 기준 local +Y 방향으로 offset pose를 만든다.

        grasp_pose:
            비전이 보낸 move_l용 최종 잡는 pose.
            [x, y, z, rx, ry, rz]

        offset_mm:
            grasp pose에서 local +Y 방향으로 떨어질 거리.
            예: 20.0이면 잡기 전 pregrasp 위치.

        반환:
            [x', y', z', rx, ry, rz]
            자세 rx, ry, rz는 비전값을 그대로 유지한다.
        """
        grasp_pose = np.asarray(grasp_pose, dtype=float).reshape(6).copy()

        R = self._pose6_to_R_zyx(grasp_pose)
        local_y_in_base = R[:, 1]

        target_pose = grasp_pose.copy()
        target_pose[:3] = grasp_pose[:3] + float(offset_mm) * local_y_in_base

        self.get_logger().info(
            f"[6D DIRECT PICK POSE] local_y_offset={float(offset_mm):.1f}mm, "
            f"local_y={np.round(local_y_in_base, 4)}, "
            f"pose={np.round(target_pose, 3)}"
        )

        return target_pose

    def _make_pose_offset_along_local_z(
        self,
        grasp_pose: np.ndarray,
        offset_mm: float,
    ) -> np.ndarray:
        """
        비전에서 받은 최종 grasp pose 기준 local +Z 방향으로 offset pose를 만든다.

        grasp_pose:
            비전이 보낸 move_l용 최종 잡는 pose.
            [x, y, z, rx, ry, rz]

        offset_mm:
            grasp pose에서 local +Z 방향으로 떨어질 거리.
            예: 50.0이면 잡은 후 상승 위치.

        반환:
            [x', y', z', rx, ry, rz]
            자세 rx, ry, rz는 비전값을 그대로 유지한다.
        """
        grasp_pose = np.asarray(grasp_pose, dtype=float).reshape(6).copy()

        R = self._pose6_to_R_zyx(grasp_pose)
        local_z_in_base = R[:, 2]

        target_pose = grasp_pose.copy()
        target_pose[:3] = grasp_pose[:3] + float(offset_mm) * local_z_in_base

        self.get_logger().info(
            f"[6D DIRECT PICK POSE] local_z_offset={float(offset_mm):.1f}mm, "
            f"local_z={np.round(local_z_in_base, 4)}, "
            f"pose={np.round(target_pose, 3)}"
        )

        return target_pose

    def make_pick_pose(self, offset_mm: float) -> np.ndarray:
        if self.ctx.current_peg_pick_pose is None:
            raise RuntimeError("No selected peg target")

        if self.use_6d_peg_interface:
            grasp_pose = self.ctx.current_peg_pick_pose.copy()

            # 새 6D 방식:
            # 비전에서 받은 pose는 이미 move_l에 넣을 최종 grasp pose이다.
            # 따라서 제어부에서는 yaw 보정, object frame 재구성, TCP 자세 보정을 하지 않는다.
            if self.ctx.current_peg_object_T is None:
                if offset_mm == self.ctx.pick_approach_above_peg_z_mm:
                    # 잡기 전 pregrasp 위치: grasp pose 기준 local +Y 방향 20mm
                    return self._make_pose_offset_along_local_y(
                        grasp_pose,
                        offset_mm=20.0,
                    )

                if offset_mm == self.ctx.pick_grasp_above_peg_z_mm:
                    # 실제 잡는 위치: 비전에서 받은 move_l pose 그대로 사용
                    self.get_logger().info(
                        f"[6D DIRECT PICK POSE] grasp pose direct = "
                        f"{np.round(grasp_pose, 3)}"
                    )
                    return grasp_pose

                if offset_mm == self.ctx.pick_lift_above_peg_z_mm:
                    # 잡고 빠져나오는 위치: grasp pose 기준 local +Y 방향 50mm
                    # pregrasp와 같은 축을 사용한다.
                    return self._make_pose_offset_along_local_y(
                        grasp_pose,
                        offset_mm=50.0,
                    )

                # 예외적으로 다른 offset이 들어오면 pregrasp/lift와 같은 local +Y offset으로 처리한다.
                return self._make_pose_offset_along_local_y(
                    grasp_pose,
                    offset_mm=float(offset_mm),
                )

            # 이전 4x4 matrix 방식 호환용.
            # 새 카메라 노드가 [x, y, z, rx, ry, rz]를 보내는 경우에는 이 분기로 들어오지 않는다.
            if self.ctx.current_target_id is None:
                raise RuntimeError("No selected peg object id")

            return self._make_tcp_pick_pose_from_object_T(
                self.ctx.current_peg_object_T,
                offset_mm,
                self.ctx.current_target_id,
            )

        pose = self.ctx.current_peg_pick_pose.copy()
        if offset_mm == self.ctx.pick_approach_above_peg_z_mm:
            pose[2] = self.ctx.pick_down_target_z_mm + self.ctx.pick_approach_offset_z_mm
        elif offset_mm == self.ctx.pick_grasp_above_peg_z_mm:
            pose[2] = self.ctx.pick_down_target_z_mm
        elif offset_mm == self.ctx.pick_lift_above_peg_z_mm:
            pose[2] = self.ctx.pick_up_target_z_mm
        else:
            pose[2] = self.ctx.pick_down_target_z_mm + float(offset_mm)
        return pose


    def make_tilted_place_pose(self) -> np.ndarray:
        """
        hole place pose를 기준으로 삽입 직전 tilt pose를 만든다.

        - 접근 MOVE_L은 기존처럼 tilt 없이 수행한다.
        - 이 함수는 place_down_target_z_mm까지 내려갈 때만 사용한다.
        - yaw 방향으로 place_tilt_deg만큼 tilt를 주기 위해 rx/ry에 분배한다.
        - 네모/object_id=1, 십자가/object_id=2는 90도 대칭이므로 place rz를 0~90도 범위로 접는다.

        식:
            rx = base_rx - tilt * sin(yaw)
            ry = base_ry + tilt * cos(yaw)

        yaw는 target_pose[5] = rz 기준이다.
        """
        if self.ctx.current_hole_place_pose is None:
            raise RuntimeError("No selected hole target")

        target_pose = self.ctx.current_hole_place_pose.copy()
        target_pose[2] = self.ctx.place_down_target_z_mm

        base_rx = float(target_pose[3])
        base_ry = float(target_pose[4])
        tilt_deg = float(self.ctx.place_tilt_deg)

        # 네모 peg/object_id=1: yaw +45 deg 보정 후 90도 대칭 범위(0~90)로 접는다.
        if self.ctx.current_target_id == 1:
            raw_yaw = float(target_pose[5])
            corrected_yaw = self._normalize_yaw_0_to_90_deg(raw_yaw + 45.0)

            target_pose[0] += 2.0
            target_pose[5] = corrected_yaw
            tilt_deg += 5.0

            self.get_logger().info(
                f"[PLACE SQUARE] apply x offset +2.0 mm, "
                f"yaw +45 then mod90: {raw_yaw:.3f} -> {target_pose[5]:.3f}, "
                f"target x={target_pose[0]:.2f}, total tilt={tilt_deg:.2f}deg"
            )

        # 십자가 peg/object_id=2: 90도 대칭 범위(0~90)로 접고,
        # yaw 방향 XY offset 및 추가 tilt를 적용한다.
        if self.ctx.current_target_id == 2:
            raw_yaw = float(target_pose[5])
            corrected_yaw = self._normalize_yaw_0_to_90_deg(raw_yaw)

            extra_tilt_deg = 7.0
            offset_mm = 10.0

            target_pose[5] = corrected_yaw
            tilt_deg += extra_tilt_deg

            yaw_rad = np.deg2rad(target_pose[5])
            target_pose[0] += offset_mm * np.cos(yaw_rad)
            target_pose[1] += offset_mm * np.sin(yaw_rad)
            target_pose[2] += 4.0

            self.get_logger().info(
                f"[PLACE CROSS] yaw mod90: {raw_yaw:.3f} -> {target_pose[5]:.3f}, "
                f"apply extra tilt +{extra_tilt_deg:.2f}deg, "
                f"xy yaw offset={offset_mm:.2f}mm, "
                f"target x={target_pose[0]:.2f}, y={target_pose[1]:.2f}, z={target_pose[2]:.2f}, "
                f"total tilt={tilt_deg:.2f}deg"
            )

        tilt_yaw_deg = float(target_pose[5])
        tilt_yaw_rad = np.deg2rad(tilt_yaw_deg)

        target_pose[3] = base_rx - tilt_deg * np.sin(tilt_yaw_rad)
        target_pose[4] = base_ry + tilt_deg * np.cos(tilt_yaw_rad)

        self.get_logger().info(
            f"[PLACE TILT] tilt={tilt_deg:.2f}deg, yaw={tilt_yaw_deg:.2f}deg, "
            f"base_rpy=[{base_rx:.2f}, {base_ry:.2f}, {tilt_yaw_deg:.2f}], "
            f"tilted_pose={np.round(target_pose, 3)}"
        )

        return target_pose

    def make_lift_pose_from_current_tcp(self) -> np.ndarray:
        """
        servo_t와 release 이후 실제 TCP pose를 읽고,
        현재 TCP의 world z만 place_lift_current_tcp_z_mm만큼 올린다.
        """
        current_tcp = self.motion.get_current_tcp_pose()
        target_pose = current_tcp.copy()
        target_pose[2] += float(self.ctx.place_lift_current_tcp_z_mm)

        self.get_logger().info(
            f"[PLACE LIFT CURRENT TCP] current_tcp={np.round(current_tcp, 3)}, "
            f"lift_z={self.ctx.place_lift_current_tcp_z_mm:.1f}mm, "
            f"target_pose={np.round(target_pose, 3)}"
        )

        return target_pose

    def save_last_pick_pose(self):
        if self.ctx.current_peg_pick_pose is None:
            raise RuntimeError("No selected peg target")

        up_pose = self.make_pick_pose(self.ctx.pick_lift_above_peg_z_mm)
        down_pose = self.make_pick_pose(self.ctx.pick_grasp_above_peg_z_mm)

        self.ctx.last_pick_up_pose = up_pose.copy()
        self.ctx.last_pick_down_pose = down_pose.copy()
        self.ctx.last_pick_id = self.ctx.current_target_id

        self.get_logger().info(
            f"[RECOVERY SAVE] last_pick_id = {self.ctx.last_pick_id}, "
            f"up_pose = {self.ctx.last_pick_up_pose}, "
            f"down_pose = {self.ctx.last_pick_down_pose}"
        )

    def set_skip_once_from_last_pick(self):
        """
        방금 matching jig가 없어서 되돌려놓은 peg의 id를
        다음 peg 선택에서 딱 1번 제외하기 위해 저장한다.
        """
        self.ctx.skip_once_pick_id = self.ctx.last_pick_id

        self.get_logger().info(
            f"[SKIP_ONCE] set skip id once = {self.ctx.skip_once_pick_id} "
            f"({self.vision.shape_name(self.ctx.skip_once_pick_id)})"
        )

    def clear_skip_once(self):
        self.ctx.skip_once_pick_pose = None
        self.ctx.skip_once_pick_id = None

    def clear_current_task(self):
        self.ctx.current_peg_index = -1
        self.ctx.current_hole_index = -1
        self.ctx.current_peg_pick_pose = None
        self.ctx.current_peg_object_T = None
        self.ctx.current_hole_place_pose = None
        self.ctx.current_target_id = None

    def clear_recovery_pose(self):
        self.ctx.last_pick_up_pose = None
        self.ctx.last_pick_down_pose = None
        self.ctx.last_pick_id = None

    def consume_current_task(self):
        if self.ctx.current_peg_index >= 0:
            if len(self.ctx.peg_targets) > self.ctx.current_peg_index:
                del self.ctx.peg_targets[self.ctx.current_peg_index]

        if self.ctx.current_hole_index >= 0:
            if len(self.ctx.active_jig_targets) > self.ctx.current_hole_index:
                used_jig = self.ctx.active_jig_targets[self.ctx.current_hole_index]
                self.get_logger().info(
                    f"[JIG] Remove used active jig slot index = "
                    f"{self.ctx.current_hole_index}, "
                    f"id = {used_jig.object_id} "
                    f"({self.vision.shape_name(used_jig.object_id)})"
                )
                del self.ctx.active_jig_targets[self.ctx.current_hole_index]

        if len(self.ctx.active_jig_targets) == 0:
            self.ctx.remaining_jig_counts.clear()
            self.ctx.hole_targets = []
            self.pause_before_next_peg_inspect = True
            self.get_logger().info(
                "[JIG] Active jig layout is fully filled. "
                "Next cycle will inspect new peg/jig positions."
            )
            self.get_logger().info(
                "[PAUSE] Pause is armed for the next peg inspection."
            )

        self.clear_current_task()
        self.clear_recovery_pose()

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def step(self):
        self.get_logger().info(f"[STATE] {self.state.name}")

        if self.state == TaskState.IDLE_HOME:
            self.motion.move_j_and_wait(self.ctx.home_joint)
            self.gripper.open()
            time.sleep(0.5)
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE:
            self.motion.move_j_and_wait(self.ctx.peg_camera_joint)
            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID:
            # J1은 경유 각도, J2~J6은 peg 카메라 자세로 먼저 정렬
            via_joint = self.ctx.peg_camera_joint.copy()
            via_joint[0] = self.ctx.home_joint[0]

            self.motion.move_j_and_wait(via_joint)
            self.motion.move_j_and_wait(self.ctx.peg_camera_joint)

            self.state = TaskState.INSPECT_PEGS

        elif self.state == TaskState.INSPECT_PEGS:
            self.wait_for_space_before_peg_inspect()
            self.ctx.peg_targets = self.vision.inspect_pegs(
                preferred_object_ids=self.get_preferred_peg_request_ids()
            )

            if len(self.ctx.peg_targets) == 0:
                self.get_logger().info("[INFO] No peg remaining")
                self.state = TaskState.RETURN_HOME
            else:
                if self.select_next_peg():
                    self.state = TaskState.MOVE_TO_TARGET_PEG
                else:
                    self.get_logger().warn(
                        "[INFO] No peg matched with remembered jig layout"
                    )
                    self.state = TaskState.RETURN_HOME

        elif self.state == TaskState.MOVE_TO_TARGET_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.make_pick_pose(self.ctx.pick_approach_above_peg_z_mm)

            # matching hole이 없을 때 원래 위치로 되돌리기 위해 저장한다.
            # 복구 시에는 pick_up_target_z_mm 높이로 먼저 돌아온 뒤,
            # move_l로 pick_down_target_z_mm까지 내려간다.
            self.save_last_pick_pose()

            # peg 잡기 전, 조금 높은 접근 위치로 이동
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
                preserve_orientation=self.use_6d_peg_interface,
            )
            self.state = TaskState.DESCEND_TO_PEG

        elif self.state == TaskState.DESCEND_TO_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.make_pick_pose(self.ctx.pick_grasp_above_peg_z_mm)

            # 그리퍼 닫기 직전, peg 잡는 높이로 하강
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
                preserve_orientation=self.use_6d_peg_interface,
            )
            self.state = TaskState.GRASP_PEG

        elif self.state == TaskState.GRASP_PEG:
            self.gripper.close()
            time.sleep(self.ctx.grasp_wait_sec)
            self.state = TaskState.LIFT_WITH_PEG

        elif self.state == TaskState.LIFT_WITH_PEG:
            if self.ctx.current_peg_pick_pose is None:
                raise RuntimeError("No selected peg target")

            target_pose = self.make_pick_pose(self.ctx.pick_lift_above_peg_z_mm)

            self.motion.move_l_and_wait(
                target_pose,
                preserve_orientation=self.use_6d_peg_interface,
            )
            self.state = TaskState.MOVE_TO_HOLE_CAMERA_POSE

        elif self.state == TaskState.MOVE_TO_HOLE_CAMERA_POSE:
            # J1은 경유 각도, J2~J6은 hole 카메라 자세로 먼저 정렬
            via_joint = self.ctx.hole_camera_joint.copy()
            via_joint[0] = self.ctx.home_joint[0]

            self.motion.move_j_and_wait(via_joint)
            self.motion.move_j_and_wait(self.ctx.hole_camera_joint)

            self.state = TaskState.INSPECT_HOLES

        elif self.state == TaskState.INSPECT_HOLES:
            if len(self.ctx.active_jig_targets) == 0:
                self.ctx.hole_targets = self.vision.inspect_holes()
                self._initialize_active_jig_layout_if_needed()
            else:
                self.get_logger().info(
                    "[JIG] Use remembered active jig layout without refreshing hole positions. "
                    f"remaining slot count = {len(self.ctx.active_jig_targets)}, "
                    f"remaining_jig_counts = {dict(self.ctx.remaining_jig_counts)}"
                )

            matched = self.select_next_hole()

            if matched:
                self.state = TaskState.MOVE_TO_TARGET_HOLE
            else:
                self.get_logger().warn(
                    "[RECOVERY] Matching hole is not found. "
                    "Return peg to original pick place through J1 midpoint."
                )
                self.state = TaskState.RETURN_TO_PICK_VIA_MID

        elif self.state == TaskState.MOVE_TO_TARGET_HOLE:
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()

            # 네모/십자가는 90도 대칭이므로 place rz를 0~90도 범위로 접는다.
            if self.ctx.current_target_id == 1:
                raw_yaw = float(target_pose[5])
                target_pose[0] += 2.0
                target_pose[5] = self._normalize_yaw_0_to_90_deg(raw_yaw + 45.0)
                self.get_logger().info(
                    f"[PLACE SQUARE APPROACH] apply x offset +2.0 mm, "
                    f"yaw +45 then mod90: {raw_yaw:.3f} -> {target_pose[5]:.3f}, "
                    f"target x = {target_pose[0]:.2f}"
                )

            elif self.ctx.current_target_id == 2:
                raw_yaw = float(target_pose[5])
                target_pose[5] = self._normalize_yaw_0_to_90_deg(raw_yaw)
                self.get_logger().info(
                    f"[PLACE CROSS APPROACH] yaw mod90: "
                    f"{raw_yaw:.3f} -> {target_pose[5]:.3f}"
                )

            target_pose[2] = self.ctx.place_approach_target_z_mm

            # hole 위 접근 위치로 이동: 기존 방식 유지, tilt 없음
            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
            )
            self.state = TaskState.MOVE_TO_TILTED_HOLE

        elif self.state == TaskState.MOVE_TO_TILTED_HOLE:
            # 삽입 직전 위치로 이동하되, yaw 방향으로 place_tilt_deg만큼 tilt 적용
            target_pose = self.make_tilted_place_pose()

            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
                preserve_orientation=True,
            )
            self.state = TaskState.SERVO_INSERT_DOWN

        elif self.state == TaskState.SERVO_INSERT_DOWN:
            # 약 2초간 servo_t로 아주 약한 하방 삽입 토크를 적용한다.
            # 실제 토크 방향/크기는 런치 파라미터 servo_insert_down_torque에서 조정한다.
            self.motion.run_servo_t_constant_torque(
                torque=self.ctx.servo_insert_down_torque,
                duration_sec=self.ctx.servo_insert_duration_sec,
                log_name="SERVO_INSERT_DOWN",
            )
            self.state = TaskState.SERVO_LEVEL_J4_J5

        elif self.state == TaskState.SERVO_LEVEL_J4_J5:
            # 업로드 코드 기반: q4_des = 90 - q2 - q3, q5_des = 90
            # 종료 조건: q4/q5 오차가 허용 오차 이하로 연속 유지되면 다음 상태로 진행
            reached = self.motion.run_servo_t_level_j4_j5_until_reached()

            if reached:
                self.get_logger().info("[SERVO_LEVEL_J4_J5] finished by target tolerance")
            else:
                self.get_logger().warn(
                    "[SERVO_LEVEL_J4_J5] finished by timeout. Continue to release."
                )

            self.state = TaskState.RELEASE_PEG

        elif self.state == TaskState.DESCEND_TO_HOLE:
            # 기존 place 방식 호환용 상태.
            # 새 place sequence에서는 MOVE_TO_TILTED_HOLE/SERVO_* 상태를 사용한다.
            if self.ctx.current_hole_place_pose is None:
                raise RuntimeError("No selected hole target")

            target_pose = self.ctx.current_hole_place_pose.copy()

            if self.ctx.current_target_id == 1:
                raw_yaw = float(target_pose[5])
                target_pose[0] += 2.0
                target_pose[5] = self._normalize_yaw_deg(raw_yaw + 45.0)
                self.get_logger().info(
                    f"[PLACE OFFSET] square descend. apply x offset +2.0 mm, "
                    f"yaw +45.0 deg: {raw_yaw:.3f} -> {target_pose[5]:.3f}, "
                    f"target x = {target_pose[0]:.2f}"
                )

            target_pose[2] = self.ctx.place_down_target_z_mm

            self.motion.move_l_and_wait(
                target_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
            )
            self.state = TaskState.RELEASE_PEG

        elif self.state == TaskState.RELEASE_PEG:
            self.gripper.open()
            time.sleep(self.ctx.release_wait_sec)

            # 정상 삽입 성공으로 보고 현재 타입의 jig 사용 count를 감소시킨다.
            self.mark_current_jig_used()

            self.state = TaskState.LIFT_FROM_HOLE

        elif self.state == TaskState.LIFT_FROM_HOLE:
            # servo_t 이후 실제 TCP 위치가 계획 pose와 달라질 수 있으므로
            # 현재 TCP를 읽은 뒤 world z만 조금 올린다.
            target_pose = self.make_lift_pose_from_current_tcp()

            self.motion.move_l_and_wait(
                target_pose,
                preserve_orientation=True,
            )
            self.state = TaskState.CHECK_REMAINING_TASK

        elif self.state == TaskState.CHECK_REMAINING_TASK:
            if len(self.ctx.remaining_jig_counts) == 0:
                self.get_logger().info(
                    "[INFO] All remembered jigs are used. "
                    "Start new cycle from first peg."
                )

            self.consume_current_task()
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE_VIA_MID

        elif self.state == TaskState.RETURN_TO_PICK_VIA_MID:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            # hole/camera 쪽에서 바로 cartesian으로 돌아가지 않고,
            # 현재 joint를 기준으로 J1만 home 쪽으로 먼저 경유한다.
            self.motion.move_j1_only_and_wait(self.ctx.home_joint[0])
            self.state = TaskState.RETURN_TO_PICK_UP_POSE

        elif self.state == TaskState.RETURN_TO_PICK_UP_POSE:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] return to original pick up pose = "
                f"{self.ctx.last_pick_up_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_up_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
                preserve_orientation=self.use_6d_peg_interface,
            )
            self.state = TaskState.DESCEND_TO_PICK_PLACE

        elif self.state == TaskState.DESCEND_TO_PICK_PLACE:
            if self.ctx.last_pick_down_pose is None:
                raise RuntimeError("No saved pick down pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] descend to original pick pose = "
                f"{self.ctx.last_pick_down_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_down_pose,
                speed=self.ctx.descend_move_l_speed,
                acc=self.ctx.descend_move_l_acc,
                preserve_orientation=self.use_6d_peg_interface,
            )
            self.state = TaskState.RELEASE_BACK_TO_PICK_PLACE

        elif self.state == TaskState.RELEASE_BACK_TO_PICK_PLACE:
            self.get_logger().info("[RECOVERY] open gripper and return peg")
            self.gripper.open()
            time.sleep(self.ctx.release_wait_sec)

            # 방금 원위치에 되돌려놓은 peg는 다음 선택에서 한 번만 제외한다.
            self.set_skip_once_from_last_pick()

            self.state = TaskState.LIFT_FROM_PICK_PLACE

        elif self.state == TaskState.LIFT_FROM_PICK_PLACE:
            if self.ctx.last_pick_up_pose is None:
                raise RuntimeError("No saved pick up pose for recovery")

            self.get_logger().info(
                f"[RECOVERY] lift after returning peg = "
                f"{self.ctx.last_pick_up_pose}"
            )

            self.motion.move_l_and_wait(
                self.ctx.last_pick_up_pose,
                speed=self.ctx.approach_move_l_speed,
                acc=self.ctx.approach_move_l_acc,
                preserve_orientation=self.use_6d_peg_interface,
            )

            self.clear_current_task()
            self.clear_recovery_pose()

            # 방금 되돌려놓은 뒤에는 home/j1 경유 없이 바로 peg 사진 자세로 이동한다.
            # remaining_jig_counts는 유지한다.
            self.state = TaskState.MOVE_TO_PEG_CAMERA_POSE

        elif self.state == TaskState.RETURN_HOME:
            self.motion.move_j_and_wait(self.ctx.home_joint)
            self.state = TaskState.DONE

        elif self.state == TaskState.DONE:
            self.get_logger().info("[DONE] Task completed")

        elif self.state == TaskState.ERROR:
            self.get_logger().error("[ERROR] Task stopped")

        else:
            raise RuntimeError(f"Unhandled state: {self.state}")

    def run(self):
        try:
            self.motion.set_operation_mode()

            while rclpy.ok() and self.state not in (TaskState.DONE, TaskState.ERROR):
                self.step()
                rclpy.spin_once(self, timeout_sec=0.01)

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            self.state = TaskState.ERROR

        finally:
            try:
                self.gripper.stop()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)

    node = PegInHoleController()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()