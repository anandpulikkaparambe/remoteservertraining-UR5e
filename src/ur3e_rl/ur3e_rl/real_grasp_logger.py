"""Structured CSV logger for real-hardware grasp attempts (2026-09-04, implementation spec
section 6). No persisted, per-attempt log existed for real-hardware runs before this --
prior real-hardware grasp results (e.g. the 2026-09-03 first confirmed grasp-and-lift) were
only ever captured as terminal/ROS log observations at the time, recorded into session notes
after the fact rather than into a file. This exists so future reach-from-distance attempts
(once curriculum-widened training produces something worth testing on real hardware) can be
compared against each other from raw data, the same way the sim side's hardware_log_*.csv
already allows.

Not wired into any existing real_phase2_*.py script by this pass -- creating the logger
itself is the requested, low-risk secondary fix; wiring it into a specific script is a
follow-up once there's an actual reach-from-distance attempt worth logging.

Usage:
    from ur3e_rl.real_grasp_logger import RealGraspLogger
    logger = RealGraspLogger()  # creates ./rl_logs/real_hw_logs/real_grasp_log_<timestamp>.csv
    logger.log_attempt(
        standoff_m=0.12, dist_at_grasp_m=0.031, object_grasped=True,
        contact_sensor_agrees=None,  # None if no independent contact signal was checked
        notes="reach-from-height test, curriculum_level~0.3 checkpoint",
    )
"""
import os
import csv
import datetime


class RealGraspLogger:
    """One row per real-hardware grasp attempt. Columns are deliberately a superset of what
    any single attempt might have available (e.g. contact_sensor_agrees is often unknown) --
    leave a field blank rather than guessing, so gaps in the data stay visible instead of
    silently defaulting to a value that looks like a real reading."""

    HEADERS = [
        "Timestamp", "Standoff_M", "Dist_At_Grasp_M", "Object_Grasped",
        "Contact_Sensor_Agrees", "Notes",
    ]

    def __init__(self, log_dir="./rl_logs/real_hw_logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"real_grasp_log_{timestamp}.csv")
        with open(self.csv_path, mode="w", newline="") as f:
            csv.writer(f).writerow(self.HEADERS)

    def log_attempt(self, standoff_m, dist_at_grasp_m, object_grasped,
                     contact_sensor_agrees=None, notes=""):
        """standoff_m: intended reach distance for this attempt (meters), not necessarily
        achieved -- record what was targeted so it can be cross-checked against the actual
        arm state if needed.
        dist_at_grasp_m: measured EE-to-target distance at the moment of the grasp attempt.
        object_grasped: the gripper driver's own ground-truth signal (e.g. Robotiq Modbus
        object_grasped, per project_phase2_rl_status memory), not an inferred/simulated value.
        contact_sensor_agrees: True/False if an independent contact signal was also checked
        this attempt and can be compared against object_grasped, else None (unknown/not
        checked) -- leave None rather than guessing.
        """
        with open(self.csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().isoformat(),
                f"{standoff_m:.4f}",
                f"{dist_at_grasp_m:.4f}",
                bool(object_grasped),
                "" if contact_sensor_agrees is None else bool(contact_sensor_agrees),
                notes,
            ])
