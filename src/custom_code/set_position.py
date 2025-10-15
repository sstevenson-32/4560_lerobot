import time
from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors

# CONFIGURATION VARIABLES
PORT_ID = "COM11"
ROBOT_NAME = "follower-1"

# --- Specified Parameters ---
'''
This is the format of the goal position dictionary used for goal position sync write.
The gripper command takes values of 0-100, while the other joints take values of degrees, based on
the settings specified in the bus initialization.
'''
desired_position = {
    'shoulder_pan': 0.0,   # degrees
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 10.0           # 0-100 range
}
move_time = 2.0  # seconds to reach desired position
hold_time = 2.0  # total time to hold at 

# ------------------------

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)
starting_pose = bus.sync_read("Present_Position")
move_to_pose(bus, desired_position, move_time)
hold_position(bus, hold_time)
move_to_pose(bus, starting_pose, move_time)
bus.disable_torque()
