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


    @staticmethod
    def wrap_deg_180(angle_deg: float) -> float:
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    @staticmethod
    def soft_deadband(err: float, deadband: float) -> float:
        err = float(err)
        deadband = float(deadband)
        if abs(err) <= deadband:
            return 0.0
        return float(np.sign(err) * (abs(err) - deadband))

    def _send_servo_t(self, torque: np.ndarray):
        torque = np.asarray(torque, dtype=float).reshape(6)

        if self.use_simulation_mode:
            self.node.get_logger().info(
                f"[SERVO_T SIM] torque = {np.round(torque, 4).tolist()}"
            )
            return

        rc = rb.ResponseCollector()
        ret = self.robot.move_servo_t(
            rc,
            [float(v) for v in torque],
            float(self.ctx.servo_t_t1_sec),
            float(self.ctx.servo_t_t2_sec),
            compensation=int(self.ctx.servo_t_compensation),
        )

        try:
            rc.error().throw_if_not_empty()
        except Exception as e:
            self.node.get_logger().warn(f"[SERVO_T] response error: {e}")

        if ret is not None and hasattr(ret, "is_success") and not ret.is_success():
            self.node.get_logger().warn(f"[SERVO_T] move_servo_t failed: {ret}")

    def send_servo_t_zero(self):
        self._send_servo_t(np.zeros(6, dtype=float))

    def run_servo_t_constant_torque(
        self,
        torque: np.ndarray,
        duration_sec: float,
        log_name: str = "SERVO_T_CONST",
    ):
        """
        duration_sec 동안 지정한 외부 토크를 servo_t로 송신한다.

        주의:
            이 함수는 joint torque를 직접 넣는다.
            Cartesian -Z force를 정확히 만들려면 Jacobian 기반 tau = J^T F가 필요하다.
            현재는 사용자가 지정한 torque vector를 매우 작게 넣는 안전한 통합 형태다.
        """
        torque = np.asarray(torque, dtype=float).reshape(6)
        duration_sec = float(duration_sec)

        self.node.get_logger().info(
            f"[{log_name}] start duration={duration_sec:.2f}s, "
            f"torque={np.round(torque, 4).tolist()}, "
            f"compensation={int(self.ctx.servo_t_compensation)}"
        )

        start_time = time.monotonic()
        last_log_time = 0.0

        try:
            while rclpy.ok():
                now = time.monotonic()
                elapsed = now - start_time

                if elapsed >= duration_sec:
                    break

                state = self.request_valid_state(retry=3, wait_sec=0.0)
                sdata = state.sdata

                if getattr(sdata, "op_stat_collision_occur", False):
                    raise RuntimeError(f"[{log_name}] Robot collision detected")

                if getattr(sdata, "op_stat_sos_flag", 0) == 4:
                    raise RuntimeError(f"[{log_name}] Command input error")

                self._send_servo_t(torque)

                if now - last_log_time > 0.2:
                    last_log_time = now
                    self.node.get_logger().info(
                        f"[{log_name}] remain={duration_sec - elapsed:.2f}s, "
                        f"torque={np.round(torque, 4).tolist()}"
                    )

                rclpy.spin_once(self.node, timeout_sec=0.0)
                time.sleep(float(self.ctx.servo_t_sleep_sec))

        finally:
            self.node.get_logger().info(f"[{log_name}] send zero external torque")
            self.send_servo_t_zero()

    def _smooth_servo_torque(
        self,
        raw_torque: np.ndarray,
        prev_torque: np.ndarray,
    ) -> np.ndarray:
        raw_torque = np.asarray(raw_torque, dtype=float).reshape(6)
        prev_torque = np.asarray(prev_torque, dtype=float).reshape(6)

        delta = raw_torque - prev_torque
        delta = np.clip(
            delta,
            -float(self.ctx.servo_torque_rate_limit),
            float(self.ctx.servo_torque_rate_limit),
        )

        rate_limited = prev_torque + delta

        alpha = float(self.ctx.servo_torque_lpf_alpha)
        filtered = alpha * rate_limited + (1.0 - alpha) * prev_torque

        return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

    def _make_level_j4_j5_torque(
        self,
        jpos: np.ndarray,
        jvel: np.ndarray,
    ):
        """
        업로드한 테스트 코드의 4/5번 조인트 자세복귀 토크를 상태머신용으로 이식.

        목표:
            q4_des = 90 - q2 - q3
            q5_des = servo_level_q5_des_deg
        """
        jpos = np.asarray(jpos, dtype=float).reshape(6)
        jvel = np.asarray(jvel, dtype=float).reshape(6)

        q2 = float(jpos[1])
        q3 = float(jpos[2])
        q4 = float(jpos[3])
        q5 = float(jpos[4])

        qvel4 = float(jvel[3])
        qvel5 = float(jvel[4])

        q4_des = 90.0 - q2 - q3
        q5_des = float(self.ctx.servo_level_q5_des_deg)

        q4_err = self.wrap_deg_180(q4_des - q4)
        q5_err = self.wrap_deg_180(q5_des - q5)

        q4_err_cmd = self.soft_deadband(q4_err, self.ctx.servo_level_j4_deadband_deg)
        q5_err_cmd = self.soft_deadband(q5_err, self.ctx.servo_level_j5_deadband_deg)

        tau4_raw = (
            float(self.ctx.servo_level_kp_j4) * q4_err_cmd
            - float(self.ctx.servo_level_kd_j4) * qvel4
        )
        tau5_raw = (
            float(self.ctx.servo_level_kp_j5) * q5_err_cmd
            - float(self.ctx.servo_level_kd_j5) * qvel5
        )

        tau4 = float(self.ctx.servo_level_j4_sign) * tau4_raw
        tau5 = float(self.ctx.servo_level_j5_sign) * tau5_raw

        tau4 = np.clip(
            tau4,
            -float(self.ctx.servo_level_max_j4_torque),
            float(self.ctx.servo_level_max_j4_torque),
        )
        tau5 = np.clip(
            tau5,
            -float(self.ctx.servo_level_max_j5_torque),
            float(self.ctx.servo_level_max_j5_torque),
        )

        target_torque = np.zeros(6, dtype=float)
        target_torque[3] = tau4
        target_torque[4] = tau5

        return target_torque, q4_des, q5_des, q4_err, q5_err, q4_err_cmd, q5_err_cmd

    def run_servo_t_level_j4_j5_until_reached(self) -> bool:
        """
        4번/5번 조인트는 기존 외부 토크로 자세 복귀한다.
        십자가(object_id=2) 삽입 중에는 마지막 조인트(J6)에만
        1 Hz sin 토크를 추가한다.

        핵심:
            - J4/J5 토크: 기존 smoothing/rate-limit 유지
            - J6 위글링 토크: smoothing 이후 target_torque[5]에 직접 대입
            → tau6가 j6_wiggle_raw를 실제로 따라가게 함

        종료 조건:
            abs(q4_err) <= servo_level_j4_tol_deg
            abs(q5_err) <= servo_level_j5_tol_deg
            위 조건이 servo_level_target_stable_count번 연속 유지되면 성공 종료.

        안전 조건:
            servo_level_max_duration_sec를 넘으면 timeout 종료 후 False 반환.
        """
        max_duration = float(self.ctx.servo_level_max_duration_sec)
        stable_required = int(self.ctx.servo_level_target_stable_count)

        self.node.get_logger().info(
            f"[SERVO_LEVEL_J4_J5] start. "
            f"tol_j4={self.ctx.servo_level_j4_tol_deg:.3f}deg, "
            f"tol_j5={self.ctx.servo_level_j5_tol_deg:.3f}deg, "
            f"stable_required={stable_required}, "
            f"max_duration={max_duration:.2f}s"
        )

        state = self.request_valid_state()
        prev_jpos = self.get_current_joint()
        prev_time = time.monotonic()

        filtered_jvel = np.zeros(6, dtype=float)
        prev_target_torque = np.zeros(6, dtype=float)

        start_time = time.monotonic()
        last_log_time = 0.0
        stable_count = 0
        reached = False

        # J6 위글링 설정
        # 안 돌면 7.0 유지, 너무 세면 5.0 / 3.0으로 낮추면 됨.
        j6_wiggle_amp = 2.1
        j6_wiggle_freq_hz = 2.0
        j6_wiggle_limit = 2.1

        try:
            while rclpy.ok():
                now = time.monotonic()
                elapsed = now - start_time
                loop_dt = now - prev_time
                prev_time = now

                if elapsed >= max_duration:
                    self.node.get_logger().warn(
                        f"[SERVO_LEVEL_J4_J5] timeout after {elapsed:.2f}s"
                    )
                    break

                if loop_dt <= 1e-9:
                    time.sleep(float(self.ctx.servo_t_sleep_sec))
                    continue

                state = self.request_valid_state(retry=3, wait_sec=0.0)
                sdata = state.sdata

                if getattr(sdata, "op_stat_collision_occur", False):
                    raise RuntimeError("[SERVO_LEVEL_J4_J5] Robot collision detected")

                if getattr(sdata, "op_stat_sos_flag", 0) == 4:
                    raise RuntimeError("[SERVO_LEVEL_J4_J5] Command input error")

                jpos = self.get_current_joint()

                jvel_raw = np.array(
                    [self.wrap_deg_180(jpos[i] - prev_jpos[i]) for i in range(6)],
                    dtype=float,
                ) / loop_dt
                prev_jpos = jpos.copy()

                alpha_v = float(self.ctx.servo_jvel_lpf_alpha)
                filtered_jvel = alpha_v * jvel_raw + (1.0 - alpha_v) * filtered_jvel
                jvel = filtered_jvel.copy()

                (
                    raw_torque,
                    q4_des,
                    q5_des,
                    q4_err,
                    q5_err,
                    q4_err_cmd,
                    q5_err_cmd,
                ) = self._make_level_j4_j5_torque(jpos, jvel)

                # ------------------------------------------------------------
                # 1) 기존 J4/J5 토크는 그대로 smoothing/rate-limit 적용
                #    여기서는 J6에 아무것도 더하지 않는다.
                # ------------------------------------------------------------
                target_torque = self._smooth_servo_torque(raw_torque, prev_target_torque)

                # prev_target_torque는 J4/J5 smoothing용으로만 유지한다.
                # J6 위글링은 아래에서 직접 덮어쓰기 때문에 prev값에 누적시키지 않는다.
                prev_target_torque = target_torque.copy()
                prev_target_torque[5] = 0.0

                # ------------------------------------------------------------
                # 2) 십자가(object_id=2)일 때만 J6 위글링을 smoothing 이후 직접 추가
                #    이렇게 해야 tau6가 j6_wiggle_raw를 실제로 따라간다.
                # ------------------------------------------------------------
                j6_wiggle_raw = 0.0
                j6_wiggle_cmd = 0.0

                if int(getattr(self.ctx, "current_target_id", -1)) in (1, 2):
                    j6_wiggle_raw = float(j6_wiggle_amp) * np.sin(
                        2.0 * np.pi * float(j6_wiggle_freq_hz) * elapsed
                    )

                    j6_wiggle_cmd = float(
                        np.clip(
                            j6_wiggle_raw,
                            -float(j6_wiggle_limit),
                            float(j6_wiggle_limit),
                        )
                    )

                    target_torque[5] = j6_wiggle_cmd

                q4_ok = abs(q4_err) <= float(self.ctx.servo_level_j4_tol_deg)
                q5_ok = abs(q5_err) <= float(self.ctx.servo_level_j5_tol_deg)

                if q4_ok and q5_ok:
                    stable_count += 1
                    if stable_count >= stable_required:
                        reached = True
                        self.node.get_logger().info(
                            f"[SERVO_LEVEL_J4_J5] target reached. "
                            f"q4_err={q4_err:.3f}, q5_err={q5_err:.3f}, "
                            f"stable={stable_count}/{stable_required}"
                        )
                        break
                else:
                    stable_count = 0

                self._send_servo_t(target_torque)

                if now - last_log_time > 0.2:
                    last_log_time = now
                    self.node.get_logger().info(
                        "[SERVO_LEVEL_J4_J5] "
                        f"remain={max_duration - elapsed:.2f}s, "
                        f"q2={jpos[1]:.2f}, q3={jpos[2]:.2f}, "
                        f"q4={jpos[3]:.2f}, q5={jpos[4]:.2f}, q6={jpos[5]:.2f}, "
                        f"q4_des={q4_des:.2f}, q5_des={q5_des:.2f}, "
                        f"q4_err={q4_err:.2f}, q5_err={q5_err:.2f}, "
                        f"q4_err_cmd={q4_err_cmd:.2f}, q5_err_cmd={q5_err_cmd:.2f}, "
                        f"tau4={target_torque[3]:.3f}, "
                        f"tau5={target_torque[4]:.3f}, "
                        f"tau6={target_torque[5]:.3f}, "
                        f"j6_wiggle_raw={j6_wiggle_raw:.3f}, "
                        f"j6_wiggle_cmd={j6_wiggle_cmd:.3f}, "
                        f"stable={stable_count}/{stable_required}"
                    )

                rclpy.spin_once(self.node, timeout_sec=0.0)
                time.sleep(float(self.ctx.servo_t_sleep_sec))

        finally:
            self.node.get_logger().info("[SERVO_LEVEL_J4_J5] send zero external torque")
            self.send_servo_t_zero()

        return reached

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