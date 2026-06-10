#!/usr/bin/env python3
"""
Keyboard trigger node for object 6D pose capture + mode switching.

Default trigger_topic is matched to mixed_pose_vision.launch.py:
  /manipulation/object_6d_trigger

Keys:
  o          : publish "object" to /detect_mode
  i          : publish "insert" to /detect_mode
  c or SPACE : publish "object" mode + "nearest" trigger
  1          : publish "object" mode + "cross"
  2          : publish "object" mode + "cylinder"
  3          : publish "object" mode + "hole"
  q or ESC   : quit
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardObjectTriggerNode(Node):
    def __init__(self):
        super().__init__("keyboard_object_trigger_node")

        self.declare_parameter("trigger_topic", "/manipulation/object_6d_trigger")
        self.declare_parameter("detect_mode_topic", "/detect_mode")
        self.declare_parameter("default_trigger_data", "nearest")

        self.trigger_topic = str(self.get_parameter("trigger_topic").value)
        self.detect_mode_topic = str(self.get_parameter("detect_mode_topic").value)
        self.default_trigger_data = str(self.get_parameter("default_trigger_data").value)

        self.trigger_pub = self.create_publisher(String, self.trigger_topic, 10)
        self.mode_pub = self.create_publisher(String, self.detect_mode_topic, 10)

        self._stdin_fd = sys.stdin.fileno()
        self._old_termios = None
        self._raw_enabled = False

        if sys.stdin.isatty():
            self._old_termios = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._raw_enabled = True
        else:
            self.get_logger().warn("stdin is not a TTY. Keyboard input may not work.")

        self.timer = self.create_timer(0.02, self._timer_callback)

        self.get_logger().info("Keyboard trigger ready.")
        self.get_logger().info(f"trigger_topic={self.trigger_topic}")
        self.get_logger().info(f"detect_mode_topic={self.detect_mode_topic}")
        self.get_logger().info(
            "Keys: [o]=object, [i]=insert, "
            "[c/SPACE]=nearest, [1]=cross, [2]=cylinder, [3]=hole, [q/ESC]=quit"
        )

    def _read_key_nonblocking(self):
        if not sys.stdin.isatty():
            return None

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None

        return sys.stdin.read(1)

    def _publish_mode(self, mode: str):
        mode = mode.strip().lower()
        if mode not in ("object", "insert"):
            self.get_logger().warn(f"invalid mode: {mode}")
            return

        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f"published mode: {self.detect_mode_topic} <- '{mode}'")

    def _publish_trigger(self, data: str, force_object_mode: bool = True):
        if force_object_mode:
            self._publish_mode("object")

        msg = String()
        msg.data = data
        self.trigger_pub.publish(msg)
        self.get_logger().info(f"published trigger: {self.trigger_topic} <- '{data}'")

    def _timer_callback(self):
        key = self._read_key_nonblocking()
        if key is None:
            return

        if key in ("o", "O"):
            self._publish_mode("object")
        elif key in ("i", "I"):
            self._publish_mode("insert")
        elif key in ("c", "C", " "):
            self._publish_trigger(self.default_trigger_data, force_object_mode=True)
        elif key == "1":
            self._publish_trigger("cross", force_object_mode=True)
        elif key == "2":
            self._publish_trigger("cylinder", force_object_mode=True)
        elif key == "3":
            self._publish_trigger("hole", force_object_mode=True)
        elif key in ("q", "Q", "\x1b"):
            self.get_logger().info("quit requested")
            rclpy.shutdown()
        else:
            self.get_logger().warn(f"unknown key: {repr(key)}")

    def destroy_node(self):
        if self._raw_enabled and self._old_termios is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardObjectTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == "__main__":
    main()