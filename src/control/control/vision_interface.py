import time

import numpy as np
import rclpy
from std_msgs.msg import Float64MultiArray

try:
    from .task_types import TaskContext, VisionTarget
except ImportError:  # direct script/debug execution support
    from task_types import TaskContext, VisionTarget


class VisionInterface:
    """
    비전 촬영 trigger publish, peg/hole target subscribe, target parsing 담당.

    기본 2D/yaw 방식:
        trigger_peg, trigger_hole = [x, y, z, rx, ry, rz]
        peg_targets, hole_targets = [x, y, yaw, id, ...]

    6D peg 방식(use_6d_peg_interface=True):
        trigger_peg = [object_id, x, y, z, rx, ry, rz]
        peg_targets success = [T00, T01, ... T33, object_id]  # len(data)==17
        peg_targets failure = [visible_id0, visible_id1, ...] # len(data)!=17

    hole은 기존 2D/yaw 방식 그대로 사용한다.
    """

    VALID_OBJECT_IDS = (0, 1, 2)
    OBJECT_ID_NAME = {
        0: "cylinder",
        1: "hole",
        2: "cross",
    }

    def __init__(
        self,
        node,
        ctx: TaskContext,
        robot_motion,
        peg_targets_topic: str,
        hole_targets_topic: str,
        trigger_peg_topic: str,
        trigger_hole_topic: str,
        camera_settle_sec: float,
        use_6d_peg_interface: bool = False,
    ):
        self.node = node
        self.ctx = ctx
        self.robot_motion = robot_motion

        self.peg_targets_topic = peg_targets_topic
        self.hole_targets_topic = hole_targets_topic
        self.trigger_peg_topic = trigger_peg_topic
        self.trigger_hole_topic = trigger_hole_topic
        self.camera_settle_sec = float(camera_settle_sec)
        self.use_6d_peg_interface = bool(use_6d_peg_interface)

        self.latest_peg_xyyawid: list[tuple[float, float, float, int]] = []
        self.latest_hole_xyyawid: list[tuple[float, float, float, int]] = []
        self.latest_peg_targets_6d: list[VisionTarget] = []
        self.latest_peg_visible_ids: list[int] = []

        self.peg_msg_received = False
        self.hole_msg_received = False

        # 6D peg 응답이 새로 들어왔는지 구분하기 위한 카운터.
        # fallback 메시지 하나를 받고 바로 실패 처리하지 않도록 사용한다.
        self.peg_msg_seq = 0

        self.trigger_peg_pub = self.node.create_publisher(
            Float64MultiArray,
            self.trigger_peg_topic,
            10,
        )

        self.trigger_hole_pub = self.node.create_publisher(
            Float64MultiArray,
            self.trigger_hole_topic,
            10,
        )

        self.peg_sub = self.node.create_subscription(
            Float64MultiArray,
            self.peg_targets_topic,
            self.peg_targets_callback,
            10,
        )

        self.hole_sub = self.node.create_subscription(
            Float64MultiArray,
            self.hole_targets_topic,
            self.hole_targets_callback,
            10,
        )

    def shape_name(self, object_id: int | None) -> str:
        if object_id is None:
            return "none"
        return self.OBJECT_ID_NAME.get(object_id, "unknown")

    # ------------------------------------------------------------------
    # transform helper for 6D peg result
    # ------------------------------------------------------------------
    def _orthonormalize_R(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0.0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    def _R_to_euler_zyx_deg(self, R: np.ndarray) -> np.ndarray:
        """
        R = Rz(rz) @ Ry(ry) @ Rx(rx) 기준으로 [rx, ry, rz] deg를 복원한다.
        sixd_pose_transform_node.py의 T_mm_to_pose6_mm_deg와 같은 규칙이다.
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

    def _T_mm_to_pose6_mm_deg(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        rpy = self._R_to_euler_zyx_deg(T[:3, :3])
        p = T[:3, 3]
        return np.array([p[0], p[1], p[2], rpy[0], rpy[1], rpy[2]], dtype=float)

    # ------------------------------------------------------------------
    # vision trigger helper
    # ------------------------------------------------------------------
    def _get_ee_pose_after_settle(self) -> np.ndarray:
        if self.camera_settle_sec > 0.0:
            time.sleep(self.camera_settle_sec)
        return self.robot_motion.get_current_tcp_pose()

    def publish_ee_pose_trigger(self, pub, label: str):
        """
        사진 촬영 trigger용으로 현재 TCP pose를 publish한다.

        publish data:
            [x, y, z, rx, ry, rz]

        단위:
            x, y, z = mm
            rx, ry, rz = deg
        """
        ee_pose = self._get_ee_pose_after_settle()

        msg = Float64MultiArray()
        msg.data = [float(v) for v in ee_pose[:6]]

        pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.05)

        self.node.get_logger().info(
            f"[VISION TRIGGER] {label} trigger published. "
            f"ee_pose = {ee_pose}"
        )

    def publish_peg_6d_trigger(self, object_id: int):
        """
        6D peg trigger.

        publish data:
            [object_id, x, y, z, rx, ry, rz]
        """
        object_id = int(object_id)
        ee_pose = self._get_ee_pose_after_settle()

        msg = Float64MultiArray()
        msg.data = [float(object_id)] + [float(v) for v in ee_pose[:6]]

        self.trigger_peg_pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.05)

        self.node.get_logger().info(
            f"[VISION TRIGGER] PEG 6D trigger published. "
            f"object_id = {object_id} ({self.shape_name(object_id)}), "
            f"data = {msg.data}"
        )

    def trigger_peg_capture(self):
        self.publish_ee_pose_trigger(self.trigger_peg_pub, "PEG")

    def trigger_hole_capture(self):
        self.publish_ee_pose_trigger(self.trigger_hole_pub, "HOLE")

    # ------------------------------------------------------------------
    # vision callback / parsing
    # ------------------------------------------------------------------
    def _parse_xyyawid_msg(
        self,
        msg: Float64MultiArray,
        label: str,
    ) -> list[tuple[float, float, float, int]]:
        data = list(msg.data)

        if len(data) % 4 != 0:
            self.node.get_logger().warn(
                f"[VISION SUB] Invalid {label} data length: {len(data)}. "
                f"Expected [x1, y1, yaw1, id1, x2, y2, yaw2, id2, ...]"
            )
            return []

        targets: list[tuple[float, float, float, int]] = []

        for i in range(0, len(data), 4):
            x = float(data[i])
            y = float(data[i + 1])
            yaw = float(data[i + 2])
            object_id = int(round(float(data[i + 3])))

            if object_id not in self.VALID_OBJECT_IDS:
                self.node.get_logger().warn(
                    f"[VISION SUB] Unknown {label} id: {object_id}. "
                    f"Expected 0=cylinder, 1=hole, 2=cross. "
                    f"This target will be ignored."
                )
                continue

            targets.append((x, y, yaw, object_id))

        return targets

    def _parse_peg_6d_msg(self, msg: Float64MultiArray) -> tuple[list[VisionTarget], list[int]]:
        data = list(msg.data)

        if len(data) == 17:
            T = np.asarray(data[:16], dtype=np.float64).reshape(4, 4)
            object_id = int(round(float(data[16])))

            if object_id not in self.VALID_OBJECT_IDS:
                self.node.get_logger().warn(
                    f"[VISION SUB] Unknown 6D peg id: {object_id}. "
                    f"Expected 0=cylinder, 1=hole, 2=cross."
                )
                return [], []

            # pose6는 디버그/fallback 용도다.
            # 실제 6D pick pose는 controller에서 base_T_object로부터
            # TCP frame을 새로 구성해서 만든다.
            pose6 = self._T_mm_to_pose6_mm_deg(T)
            target = VisionTarget(
                pose=pose6,
                object_id=object_id,
                transform=T.copy(),
            )

            self.node.get_logger().info(
                f"[VISION SUB] 6D peg object frame received: "
                f"id={object_id} ({self.shape_name(object_id)}), "
                f"object_pose6_debug={pose6}, "
                f"base_T_object={np.round(T, 3).tolist()}"
            )
            return [target], []

        visible_ids: list[int] = []
        for value in data:
            object_id = int(round(float(value)))
            if object_id in self.VALID_OBJECT_IDS and object_id not in visible_ids:
                visible_ids.append(object_id)

        self.node.get_logger().warn(
            f"[VISION SUB] 6D peg pose unavailable. "
            f"data_len={len(data)}, visible_ids={visible_ids}, raw_data={data}"
        )
        return [], visible_ids

    def peg_targets_callback(self, msg: Float64MultiArray):
        if self.use_6d_peg_interface:
            self.latest_peg_targets_6d, self.latest_peg_visible_ids = self._parse_peg_6d_msg(msg)
        else:
            self.latest_peg_xyyawid = self._parse_xyyawid_msg(msg, "peg")

        self.peg_msg_received = True
        self.peg_msg_seq += 1

        if self.use_6d_peg_interface:
            self.node.get_logger().info(
                f"[VISION SUB] peg 6D response received: "
                f"target_count={len(self.latest_peg_targets_6d)}, "
                f"visible_ids={self.latest_peg_visible_ids}"
            )
        else:
            self.node.get_logger().info(
                f"[VISION SUB] peg targets received: {len(self.latest_peg_xyyawid)}"
            )

    def hole_targets_callback(self, msg: Float64MultiArray):
        self.latest_hole_xyyawid = self._parse_xyyawid_msg(msg, "hole")
        self.hole_msg_received = True

        self.node.get_logger().info(
            f"[VISION SUB] hole targets received: {len(self.latest_hole_xyyawid)}"
        )

    def _wait_for_peg_msg(self, reset: bool = True) -> bool:
        if reset:
            self.peg_msg_received = False
            self.latest_peg_xyyawid = []
            self.latest_peg_targets_6d = []
            self.latest_peg_visible_ids = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.peg_msg_received:
                return True

        return False

    def _wait_for_hole_msg(self, reset: bool = True) -> bool:
        if reset:
            self.hole_msg_received = False
            self.latest_hole_xyyawid = []

        start_time = time.monotonic()

        while rclpy.ok():
            if time.monotonic() - start_time > self.ctx.vision_wait_timeout_sec:
                return False

            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.hole_msg_received:
                return True

        return False

    def _wait_for_peg_6d_target(self, request_id: int) -> list[VisionTarget]:
        """
        6D peg 전용 대기 함수.

        기존 _wait_for_peg_msg()는 메시지 하나만 받으면 바로 종료한다.
        6D 방식에서는 len(data)!=17 fallback visible_ids가 먼저 들어올 수 있으므로,
        성공 pose가 들어오거나 timeout이 날 때까지 계속 기다린다.

        반환:
            [VisionTarget] : request_id에 해당하는 6D pose 수신 성공
            []             : timeout 또는 해당 id pose 없음
        """
        request_id = int(request_id)

        # FoundationPose는 object 종류에 따라 3초 이상 걸릴 수 있어서
        # 6D peg는 최소 6초는 기다리도록 한다.
        timeout_sec = max(float(self.ctx.vision_wait_timeout_sec), 6.0)

        start_time = time.monotonic()
        last_seq = self.peg_msg_seq
        last_visible_ids: list[int] = []
        printed_wait_visible = False
        printed_wait_no_visible = False

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                self.node.get_logger().warn(
                    f"[VISION] 6D peg target wait timeout. "
                    f"request_id={request_id}, "
                    f"last_visible_ids={last_visible_ids}, "
                    f"timeout_sec={timeout_sec:.2f}"
                )
                return []

            rclpy.spin_once(self.node, timeout_sec=0.05)

            # 새 메시지가 안 들어왔으면 계속 대기
            if self.peg_msg_seq == last_seq:
                continue

            last_seq = self.peg_msg_seq

            # 성공 pose가 온 경우
            if len(self.latest_peg_targets_6d) > 0:
                matched_targets = [
                    target
                    for target in self.latest_peg_targets_6d
                    if int(target.object_id) == request_id
                ]

                if len(matched_targets) > 0:
                    self.node.get_logger().info(
                        f"[VISION] 6D peg request success. "
                        f"request_id={request_id}, "
                        f"target_count={len(matched_targets)}"
                    )
                    return matched_targets

                received_ids = [
                    int(target.object_id)
                    for target in self.latest_peg_targets_6d
                ]

                self.node.get_logger().warn(
                    f"[VISION] 6D peg pose received, but id mismatch. "
                    f"request_id={request_id}, received_ids={received_ids}. "
                    "Keep waiting..."
                )
                continue

            # fallback visible_ids가 온 경우
            last_visible_ids = list(self.latest_peg_visible_ids)

            if request_id in last_visible_ids:
                # 현재 id가 보이기는 하지만 pose 계산이 아직 끝나지 않은 상태로 보고 계속 대기
                if not printed_wait_visible:
                    self.node.get_logger().info(
                        f"[VISION] Requested 6D peg id={request_id} "
                        f"is visible but pose is not ready yet. "
                        f"visible_ids={last_visible_ids}. Keep waiting..."
                    )
                    printed_wait_visible = True
                continue

            if len(last_visible_ids) > 0:
                # 다른 id만 보이는 경우도 바로 실패 처리하지 않는다.
                # 이전 요청의 fallback이 늦게 들어올 수 있고,
                # FoundationPose가 현재 요청을 처리 중일 수 있으므로 timeout까지 기다린다.
                self.node.get_logger().info(
                    f"[VISION] Requested 6D peg id={request_id} pose not ready. "
                    f"visible_ids={last_visible_ids}. Keep waiting..."
                )
                continue

            if not printed_wait_no_visible:
                self.node.get_logger().info(
                    f"[VISION] Requested 6D peg id={request_id} "
                    "pose not ready. No visible object ids returned yet. Keep waiting..."
                )
                printed_wait_no_visible = True

        return []

    def _xyyaw_to_tcp_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        object_id: int,
        target_kind: str = "peg",
    ) -> np.ndarray:
        """
        [x, y, yaw, id] -> [x, y, z, rx, ry, rz]

        yaw는 object_id와 target_kind에 따라 보정한 뒤 rz에 넣는다.
        z는 이후 상태머신에서 작업 높이에 맞게 덮어쓴다.

        target_kind:
            "peg"  : peg를 잡을 때 사용하는 yaw 계산식
            "hole" : hole에 놓을 때 사용하는 yaw 계산식
        """
        corrected_yaw = self._correct_yaw_by_object_id(
            yaw=yaw,
            object_id=object_id,
            target_kind=target_kind,
        )

        return np.array(
            [
                x,
                y,
                0.0,
                self.ctx.flat_tcp_rx_deg,
                self.ctx.flat_tcp_ry_deg,
                corrected_yaw,
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # vision inspect
    # ------------------------------------------------------------------
    def inspect_pegs(self, preferred_object_ids: list[int] | None = None) -> list[VisionTarget]:
        self.node.get_logger().info("[VISION] Trigger peg capture and wait for peg targets...")

        if self.use_6d_peg_interface:
            return self.inspect_pegs_6d(preferred_object_ids)

        # 중요:
        # trigger 직후 빠르게 결과가 들어올 수 있으므로,
        # wait 함수 내부에서 다시 reset하지 않도록 여기서 먼저 초기화한다.
        self.peg_msg_received = False
        self.latest_peg_xyyawid = []

        self.trigger_peg_capture()

        if not self._wait_for_peg_msg(reset=False):
            self.node.get_logger().warn("[VISION] Peg target wait timeout")
            return []

        peg_candidates = [
            VisionTarget(
                pose=self._xyyaw_to_tcp_pose(
                    x,
                    y,
                    yaw,
                    object_id,
                    target_kind="peg",
                ),
                object_id=object_id,
            )
            for x, y, yaw, object_id in self.latest_peg_xyyawid
        ]

        for i, target in enumerate(peg_candidates):
            self.node.get_logger().info(
                f"[VISION] peg[{i}] id = {target.object_id} "
                f"({self.shape_name(target.object_id)}), "
                f"pose = {target.pose}"
            )

        self.node.get_logger().info(f"[VISION] detected peg count = {len(peg_candidates)}")
        return peg_candidates

    def inspect_pegs_6d(self, preferred_object_ids: list[int] | None = None) -> list[VisionTarget]:
        """
        6D peg 방식에서는 trigger 1회당 요청한 object_id 1개의 pose만 온다.
        len(data)==17이면 성공 pose이고, len(data)!=17이면 현재 보이는 id 목록이다.

        수정 내용:
            - fallback visible_ids 메시지를 받았다고 바로 실패 처리하지 않는다.
            - request_id에 해당하는 len(data)==17 pose가 올 때까지 기다린다.
            - timeout은 6D FoundationPose 처리 시간을 고려해서 최소 6초로 둔다.
        """
        if preferred_object_ids is None or len(preferred_object_ids) == 0:
            request_ids = list(self.VALID_OBJECT_IDS)
        else:
            request_ids = []
            for object_id in preferred_object_ids:
                object_id = int(object_id)
                if object_id in self.VALID_OBJECT_IDS and object_id not in request_ids:
                    request_ids.append(object_id)

        if len(request_ids) == 0:
            self.node.get_logger().warn("[VISION] No valid 6D peg request ids")
            return []

        for request_id in request_ids:
            self.peg_msg_received = False
            self.latest_peg_targets_6d = []
            self.latest_peg_visible_ids = []

            self.publish_peg_6d_trigger(request_id)

            targets = self._wait_for_peg_6d_target(request_id)

            if len(targets) > 0:
                return targets

            self.node.get_logger().warn(
                f"[VISION] Requested 6D peg id={request_id} "
                f"({self.shape_name(request_id)}) unavailable after waiting."
            )

        self.node.get_logger().warn(
            f"[VISION] No 6D peg pose available for request_ids={request_ids}"
        )
        return []

    def inspect_holes(self) -> list[VisionTarget]:
        self.node.get_logger().info("[VISION] Trigger hole capture and wait for hole targets...")

        # 중요:
        # trigger 직후 빠르게 결과가 들어올 수 있으므로,
        # wait 함수 내부에서 다시 reset하지 않도록 여기서 먼저 초기화한다.
        self.hole_msg_received = False
        self.latest_hole_xyyawid = []

        self.trigger_hole_capture()

        if not self._wait_for_hole_msg(reset=False):
            self.node.get_logger().warn("[VISION] Hole target wait timeout")
            return []

        hole_candidates = [
            VisionTarget(
                pose=self._xyyaw_to_tcp_pose(
                    x,
                    y,
                    yaw,
                    object_id,
                    target_kind="hole",
                ),
                object_id=object_id,
            )
            for x, y, yaw, object_id in self.latest_hole_xyyawid
        ]

        for i, target in enumerate(hole_candidates):
            self.node.get_logger().info(
                f"[VISION] hole[{i}] id = {target.object_id} "
                f"({self.shape_name(target.object_id)}), "
                f"pose = {target.pose}"
            )

        self.node.get_logger().info(f"[VISION] detected hole count = {len(hole_candidates)}")
        return hole_candidates

    def _normalize_yaw_deg(self, yaw: float) -> float:
        """
        yaw를 -180 ~ 180 deg 범위로 정규화한다.
        """
        return (float(yaw) + 180.0) % 360.0 - 180.0

    def _correct_yaw_by_object_id(
        self,
        yaw: float,
        object_id: int,
        target_kind: str = "peg",
    ) -> float:
        """
        vision에서 받은 yaw를 object_id와 target_kind에 따라 보정한다.

        id 의미:
            0: 원
            1: 사각형/insert hole class
            2: 십자가

        target_kind:
            "peg"  : peg를 잡을 때 사용하는 yaw 계산식
            "hole" : hole에 놓을 때 사용하는 yaw 계산식

        중요:
            peg 계산식과 hole 계산식은 서로 독립이다.
            따라서 peg가 잘 맞는 상태에서 hole 계산식만 수정해도
            peg 잡는 yaw에는 영향이 없다.
        """
        yaw = float(yaw)

        # ------------------------------------------------------------
        # 1. peg 잡을 때 yaw 계산식
        # ------------------------------------------------------------
        if target_kind == "peg":
            if object_id == 0:
                corrected_yaw = 135.0

            elif object_id == 1:
                corrected_yaw = (yaw % 90.0) + 135.0

            elif object_id == 2:
                corrected_yaw = (yaw % 90.0) + 135.0

            else:
                self.node.get_logger().warn(
                    f"[VISION] Unknown peg object id for yaw correction: {object_id}. "
                    f"Use raw yaw = {yaw}"
                )
                corrected_yaw = yaw

            corrected_yaw = self._normalize_yaw_deg(corrected_yaw)

            self.node.get_logger().info(
                f"[VISION] peg yaw correction: "
                f"id={object_id}, raw_yaw={yaw:.3f}, "
                f"corrected_yaw={corrected_yaw:.3f}"
            )

            return corrected_yaw

        # ------------------------------------------------------------
        # 2. hole에 놓을 때 yaw 계산식
        # ------------------------------------------------------------
        if target_kind == "hole":
            if object_id == 0:
                # 원통 hole: yaw 의미가 작으므로 필요하면 고정값 사용
                corrected_yaw = -45.0

            elif object_id == 1:
                # 사각형 hole:
                # 여기에 성현님이 원하는 hole 전용 yaw 계산식을 넣으면 됨.
                corrected_yaw = (yaw % 90.0)

            elif object_id == 2:
                # 십자가 hole:
                # 여기에 성현님이 원하는 hole 전용 yaw 계산식을 넣으면 됨.
                corrected_yaw = (yaw % 90.0) - 45.0

            else:
                self.node.get_logger().warn(
                    f"[VISION] Unknown hole object id for yaw correction: {object_id}. "
                    f"Use raw yaw = {yaw}"
                )
                corrected_yaw = yaw

            corrected_yaw = self._normalize_yaw_deg(corrected_yaw)

            self.node.get_logger().info(
                f"[VISION] hole yaw correction: "
                f"id={object_id}, raw_yaw={yaw:.3f}, "
                f"corrected_yaw={corrected_yaw:.3f}"
            )

            return corrected_yaw

        # ------------------------------------------------------------
        # 3. target_kind가 잘못 들어온 경우
        # ------------------------------------------------------------
        self.node.get_logger().warn(
            f"[VISION] Unknown target_kind for yaw correction: {target_kind}. "
            f"Use raw yaw = {yaw}"
        )
        return self._normalize_yaw_deg(yaw)