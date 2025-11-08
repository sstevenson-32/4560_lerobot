from so101_utils import load_calibration, move_to_pose, hold_position, setup_motors, pick_up_block_cubic, place_block_cubic

# CONFIGURATION VARIABLES
PORT_ID = "COM7"
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

y_offset = 0.03
x_offset = -0.01


block_one_start = [0.25, -0.1, 0.0]
block_one_target = [0.15, 0.2, 0.0]

block_two_start = [0.15, -0.15, 0.0]
block_two_target = [0.15, 0.2, 0.04]

# Add required offsets
blocks = [block_one_start, block_two_start]
targets = [block_one_target, block_two_target]
for start, end in zip(blocks, targets):
    if (start[0] > 0): start[0] -= x_offset
    elif (start[0] < 0): start[0] += x_offset

    if (start[1] > 0): start[1] -= y_offset
    elif (start[1] < 0): start[1] += y_offset

    if (end[0] > 0): end[0] -= x_offset
    elif (end[0] < 0): end[0] += x_offset

    if (end[1] > 0): end[1] -= y_offset
    elif (end[1] < 0): end[1] += y_offset

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
pick_up_block_cubic(bus, block_one_start, move_time)
hold_position(bus, hold_time)
place_block_cubic(bus, block_one_target, move_time)
hold_position(bus, hold_time)

# Pick and place second cube
pick_up_block_cubic(bus, block_two_start, move_time)
hold_position(bus, hold_time)
place_block_cubic(bus, block_two_target, move_time)
hold_position(bus, hold_time)

# End at starting config
move_to_pose(bus, starting_pose, move_time)
bus.disable_torque()
