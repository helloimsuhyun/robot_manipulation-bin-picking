import time

import numpy as np
import rbpodo as rb
import rclpy

try:
    from .task_types import TaskContext
except ImportError:  # direct script/debug execution support
    from task_types import TaskContext


class RobotMotion:
    """
    rbpodo 로봇 연결, 상태 읽기, MoveJ/MoveL 실행 및 완료 대기 담당.
    """

    def __init__(
        self,
        node,
        ctx: TaskContext,
        robot_ip: str,
        use_simulation_mode: bool,
    ):
        self.node = node
        self.ctx = ctx
        self.use_simulation_mode = bool(use_simulation_mode)

        # 명령 전송용
        self.robot = rb.Cobot(robot_ip)

        # 상태 읽기용
        self.robot_data = rb.CobotData(robot_ip)

    def set_operation_mode(self):
        rc = rb.ResponseCollector()

        if self.use_simulation_mode:
            self.robot.set_operation_mode(rc, rb.OperationMode.Simulation)
            mode_name = "Simulation"
        else:
            self.robot.set_operation_mode(rc, rb.OperationMode.Real)
            mode_name = "Real"

        rc.error().throw_if_not_empty()
        self.node.get_logger().info(f"Operation mode set to {mode_name}")

    def request_valid_state(
        self,
        retry: int = 30,
        wait_sec: float = 0.05,
    ):
        """
        robot_data.request_data()가 None을 반환할 수 있으므로 여러 번 재시도.
        """
        for _ in range(retry):
            state = self.robot_data.request_data()

            if state is not None:
                return state

            time.sleep(wait_sec)

        raise RuntimeError("robot_data.request_data() returned None repeatedly")

    @staticmethod
    def angle_abs_error_deg(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        각도 오차를 -180~180 기준으로 정규화한 뒤 절댓값으로 반환.
        """
        current = np.array(current, dtype=float)
        target = np.array(target, dtype=float)

        return np.abs((current - target + 180.0) % 360.0 - 180.0)

    def get_current_joint(self) -> np.ndarray:
        """
        현재 로봇의 j1~j6 관절각을 읽어옴.

        반환:
            np.ndarray([j1, j2, j3, j4, j5, j6])

        단위:
            degree
        """
        state = self.request_valid_state()
        sdata = state.sdata

        if not hasattr(sdata, "jnt_ang"):
            raise RuntimeError(
                f"sdata has no attribute 'jnt_ang'. "
                f"Available fields: {dir(sdata)}"
            )

        current_joint = np.array(sdata.jnt_ang, dtype=float)

        if current_joint.shape[0] < 6:
            raise RuntimeError(f"Invalid jnt_ang length: {current_joint.shape[0]}")

        return current_joint[:6]

    def get_current_tcp_pose(self) -> np.ndarray:
        """
        현재 TCP pose를 CobotData 상태값에서 읽어옴.

        반환:
            np.ndarray([x, y, z, rx, ry, rz])

        단위:
            x, y, z = mm
            rx, ry, rz = deg

        주의:
            robot.get_tcp_info(rc)는 사용하지 않음.
            태블릿/컨트롤러에 print(get_tcp_info(), ...)가 뜨는 문제를 피하기 위해
            robot_data.request_data() 기반으로 읽음.
        """
        state = self.request_valid_state()
        sdata = state.sdata

        if hasattr(sdata, "tcp"):
            tcp_info = np.array(sdata.tcp, dtype=float)

        elif hasattr(sdata, "tcp_pos"):
            tcp_info = np.array(sdata.tcp_pos, dtype=float)

        elif hasattr(sdata, "cur_pos"):
            tcp_info = np.array(sdata.cur_pos, dtype=float)

        elif hasattr(sdata, "tcp_info"):
            tcp_info = np.array(sdata.tcp_info, dtype=float)

        else:
            raise RuntimeError(
                "TCP pose field를 찾지 못했습니다. "
                f"Available fields: {dir(sdata)}"
            )

        if tcp_info.shape[0] < 6:
            raise RuntimeError(f"Invalid TCP pose length: {tcp_info.shape[0]}")

        return tcp_info[:6]

    def force_flat_gripper_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        x, y, z는 유지한다.
        rx, ry는 수평 자세로 강제한다.
        rz는 vision yaw 보정값이므로 유지한다.

        6D peg grasp에서는 이 함수를 적용하지 않고,
        비전에서 계산된 rx, ry, rz를 그대로 사용한다.
        """
        pose = np.array(pose, dtype=float).copy()

        pose[3] = self.ctx.flat_tcp_rx_deg
        pose[4] = self.ctx.flat_tcp_ry_deg

        return pose

    def wait_until_joint_reached(self, target_joint: np.ndarray) -> bool:
        """
        move_j 이후 현재 joint 값을 직접 읽어서 목표 joint에 도달했는지 판단.

        완료 조건:
            모든 관절 오차가 joint_tol_deg 이하인 상태가
            joint_stable_count_required번 연속 유지되면 완료로 판단.
        """
        target_joint = np.array(target_joint, dtype=float)

        start_time = time.monotonic()
        stable_count = 0

        while rclpy.ok():
            current_joint = self.get_current_joint()
            joint_error = self.angle_abs_error_deg(current_joint, target_joint)
            """
            self.node.get_logger().info(
                f"[WAIT_JOINT] current = {current_joint}, "
                f"target = {target_joint}, "
                f"error = {joint_error}, "
                f"stable = {stable_count}/{self.ctx.joint_stable_count_required}"
            )
            """

            if np.all(joint_error <= self.ctx.joint_tol_deg):
                stable_count += 1

                if stable_count >= self.ctx.joint_stable_count_required:
                    self.node.get_logger().info("[WAIT_JOINT] target reached")
                    return True
            else:
                stable_count = 0

            elapsed = time.monotonic() - start_time

            if elapsed > self.ctx.joint_wait_timeout_sec:
                self.node.get_logger().error(
                    f"[WAIT_JOINT] timeout after {elapsed:.2f} sec. "
                    f"current = {current_joint}, "
                    f"target = {target_joint}, "
                    f"error = {joint_error}"
                )
                return False

            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(self.ctx.joint_polling_dt_sec)

        return False

    def wait_until_tcp_reached(self, target_pose: np.ndarray) -> bool:
        """
        move_l 이후 현재 TCP pose를 직접 읽어서 목표 TCP pose에 도달했는지 판단.

        완료 조건:
            position error <= tcp_pos_tol_mm
            rotation error <= tcp_rot_tol_deg
            위 조건이 tcp_stable_count_required번 연속 유지되면 완료.
        """
        target_pose = np.array(target_pose, dtype=float)

        start_time = time.monotonic()
        stable_count = 0

        while rclpy.ok():
            current_pose = self.get_current_tcp_pose()

            pos_error = np.abs(current_pose[:3] - target_pose[:3])
            rot_error = self.angle_abs_error_deg(current_pose[3:6], target_pose[3:6])

            self.node.get_logger().info(
                f"[WAIT_TCP] current = {current_pose}, "
                f"target = {target_pose}, "
                f"pos_error = {pos_error}, "
                f"rot_error = {rot_error}, "
                f"stable = {stable_count}/{self.ctx.tcp_stable_count_required}"
            )

            pos_ok = np.all(pos_error <= self.ctx.tcp_pos_tol_mm)
            rot_ok = np.all(rot_error <= self.ctx.tcp_rot_tol_deg)

            if pos_ok and rot_ok:
                stable_count += 1

                if stable_count >= self.ctx.tcp_stable_count_required:
                    self.node.get_logger().info("[WAIT_TCP] target reached")
                    return True
            else:
                stable_count = 0

            elapsed = time.monotonic() - start_time

            if elapsed > self.ctx.tcp_wait_timeout_sec:
                self.node.get_logger().error(
                    f"[WAIT_TCP] timeout after {elapsed:.2f} sec. "
                    f"current = {current_pose}, "
                    f"target = {target_pose}, "
                    f"pos_error = {pos_error}, "
                    f"rot_error = {rot_error}"
                )
                return False

            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(self.ctx.tcp_polling_dt_sec)

        return False

    def move_j_and_wait(
        self,
        joint: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
    ):
        """
        rbpodo 기본 move_j 명령은 그대로 사용하고,
        완료 판단은 현재 joint polling 방식으로 수행.
        """
        if speed is None:
            speed = self.ctx.move_j_speed
        if acc is None:
            acc = self.ctx.move_j_acc

        rc = rb.ResponseCollector()

        joint = np.array(joint, dtype=float)

        self.node.get_logger().info(f"[MOVE_J] target joint = {joint}")

        self.robot.move_j(rc, joint, speed, acc)
        rc.error().throw_if_not_empty()

        reached = self.wait_until_joint_reached(joint)

        if not reached:
            raise RuntimeError(f"MoveJ target not reached: {joint}")

        rc.error().throw_if_not_empty()

        self.node.get_logger().info("[MOVE_J] finished by joint polling")

    def move_l_and_wait(
        self,
        pose: np.ndarray,
        speed: float | None = None,
        acc: float | None = None,
        preserve_orientation: bool = False,
    ):
        """
        rbpodo Python API 기준 move_l 사용.

        move_l signature:
            move_l(rc, point, speed, acceleration, timeout=-1.0, return_on_err=False)

        point:
            [x, y, z, rx, ry, rz]

        단위:
            x, y, z = mm
            rx, ry, rz = deg

        중요:
            preserve_orientation=False이면 기존 방식대로 rx, ry는 flat_tcp_* 값으로 강제한다.
            preserve_orientation=True이면 입력 pose의 rx, ry, rz를 그대로 사용한다.
            6D peg grasp에서는 preserve_orientation=True를 사용한다.
        """
        if speed is None:
            speed = self.ctx.move_l_speed
        if acc is None:
            acc = self.ctx.move_l_acc

        rc = rb.ResponseCollector()

        raw_pose = np.array(pose, dtype=float)
        if preserve_orientation:
            pose = raw_pose.copy()
            pose_mode = "preserve"
        else:
            pose = self.force_flat_gripper_pose(raw_pose)
            pose_mode = "flat"

        self.node.get_logger().info(f"[MOVE_L] raw target pose = {raw_pose}")
        self.node.get_logger().info(f"[MOVE_L] {pose_mode} target pose = {pose}")
        self.node.get_logger().info(f"[MOVE_L] speed = {speed}, acc = {acc}")

        try:
            current_pose = self.get_current_tcp_pose()
            self.node.get_logger().info(f"[MOVE_L DEBUG] current tcp = {current_pose}")
            self.node.get_logger().info(f"[MOVE_L DEBUG] target tcp  = {pose}")
            self.node.get_logger().info(f"[MOVE_L DEBUG] delta tcp   = {pose - current_pose}")
        except Exception as e:
            self.node.get_logger().warn(f"[MOVE_L DEBUG] current tcp read failed: {e}")

        # rb.ReferenceFrame.Base 넣지 않음.
        # rbpodo Python API의 5번째 인자는 ReferenceFrame이 아니라 timeout임.
        self.robot.move_l(rc, pose, speed, acc)
        rc.error().throw_if_not_empty()

        reached = self.wait_until_tcp_reached(pose)

        if not reached:
            raise RuntimeError(f"MoveL target not reached: {pose}")

        rc.error().throw_if_not_empty()

        self.node.get_logger().info("[MOVE_L] finished by TCP polling")

    def move_j1_only_and_wait(
        self,
        target_j1_deg: float,
        speed: float | None = None,
        acc: float | None = None,
    ):
        """
        현재 관절각을 저장한 뒤 j2~j6는 그대로 두고 j1만 target_j1_deg로 이동.
        """
        saved_joint = self.get_current_joint()

        target_joint = saved_joint.copy()
        target_joint[0] = float(target_j1_deg)

        self.node.get_logger().info(f"[MOVE_J1_ONLY] saved joint = {saved_joint}")
        self.node.get_logger().info(f"[MOVE_J1_ONLY] target joint = {target_joint}")

        self.move_j_and_wait(target_joint, speed=speed, acc=acc)
