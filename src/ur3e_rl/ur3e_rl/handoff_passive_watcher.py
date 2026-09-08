"""Passive /rosout watcher for handoff root-cause investigation (2026-09-06). Subscribes
only -- sends nothing, calls no services, never competes with training's own MoveGroup
goals. Logs every move_group-authored message (ideally at DEBUG verbosity -- see the
launch command note) to a timestamped file, so real planning-failure detail from actual
training-driven handoff attempts can be reviewed later without having disturbed the run
that produced it.

Run this in parallel with training, in its own terminal:
    ros2 run ur3e_rl handoff_passive_watcher
"""
import time
import datetime
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Log


LEVEL_NAMES = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "FATAL"}


class PassiveWatcher(Node):
    def __init__(self, out_path):
        super().__init__('handoff_passive_watcher')
        self.out_path = out_path
        self.create_subscription(Log, '/rosout', self._on_log, 200)
        self.count = 0
        self.get_logger().info(f"Passive watcher up, writing to {out_path}")

    def _on_log(self, msg):
        # 'move_group' catches move_group's own log lines; 'ur3e_rl_env_node' catches this
        # project's own new error_code logging in moveit_bridge.py's _execute_move() -- that
        # line is logged via the training env's node logger, not move_group's, so it needs
        # its own name match here or it would silently never be captured.
        if 'move_group' not in msg.name and 'ur3e_rl_env_node' not in msg.name:
            return
        self.count += 1
        level = LEVEL_NAMES.get(msg.level, str(msg.level))
        line = f"[{datetime.datetime.now().isoformat()}] [{level}] [{msg.name}] {msg.msg}\n"
        with open(self.out_path, 'a') as f:
            f.write(line)


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"./rl_logs/handoff_passive_watch_{timestamp}.log"
    if not rclpy.ok():
        rclpy.init()
    node = PassiveWatcher(out_path)
    print(f"Watching /rosout for move_group messages -> {out_path}", flush=True)
    print("Sends nothing, calls nothing -- safe to run alongside training. Ctrl+C to stop.",
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nCaptured {node.count} move_group log lines -> {out_path}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
