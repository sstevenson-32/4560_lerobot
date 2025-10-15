import time
from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors

# CONFIGURATION VARIABLES
PORT_ID = "COM6"
ROBOT_NAME = "follower-1"

# --- Specified Parameters ---

starting_configuration = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -60.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50
}

starting_configuration_closed = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -60.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

intermediate_one = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': -90.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

intermediate_two = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': -90.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5
}

intermediate_three = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -90.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50   
}

final_configuration_closed = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -60.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5    
}

final_configuration = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 0.0,
    'elbow_flex': -60.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50       
}

move_time = 2.0  # seconds to reach desired position
hold_time = 2.0  # total time to hold at 

# ------------------------

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)
starting_pose = bus.sync_read("Present_Position")
move_to_pose(bus, starting_configuration, move_time)
hold_position(bus, hold_time)

move_to_pose(bus, starting_configuration_closed, move_time)
move_to_pose(bus, intermediate_one, move_time)
move_to_pose(bus, intermediate_two, move_time)
move_to_pose(bus, final_configuration_closed, move_time)
move_to_pose(bus, final_configuration, move_time)
move_to_pose(bus, intermediate_three, move_time)

move_to_pose(bus, starting_pose, move_time)
bus.disable_torque()
