from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors, pick_up_block, place_block

# CONFIGURATION VARIABLES
PORT_ID = "COM4"
ROBOT_NAME = "follower-1"

# --- Specified Parameters ---

zero_config = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0
}

block_one_start = [0.2-0.04, -0.15+0.05, 0.0]
block_one_target = [0.25-0.05, 0.2-0.08, 0.0]

block_two_start = [0.25-0.05, -0.1+0.03, 0.0]
block_two_target = [0.25-0.05, 0.2-0.08, 0.025]

move_time = 1.5
hold_time = 0.05

# ------------------------

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# Register starting position
starting_pose = bus.sync_read("Present_Position")

# Move to zero config
move_to_pose(bus, zero_config, move_time)
hold_position(bus, hold_time)

# Pick and place first cube
pick_up_block(bus, block_one_start, move_time)
hold_position(bus, hold_time)
place_block(bus, block_one_target, move_time)
hold_position(bus, hold_time)

# Pick and place second cube
pick_up_block(bus, block_two_start, move_time)
hold_position(bus, hold_time)
place_block(bus, block_two_target, move_time)
hold_position(bus, hold_time)

# End at starting config
move_to_pose(bus, starting_pose, move_time)
bus.disable_torque()
