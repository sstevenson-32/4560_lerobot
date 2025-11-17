from so101_utils import load_calibration, move_to_pose_cubic, hold_position, setup_motors, pick_up_block_cubic, place_block_cubic


# CONFIGURATION VARIABLES
PORT_ID = "COM4"
ROBOT_NAME = "follower-1"

# --- Specified Parameters ---

zero_config = {
    'shoulder_pan': -5,
    'shoulder_lift':-7,
    'elbow_flex': -100,
    'wrist_flex': 70,
    'wrist_roll': 90,
    'gripper': 10
}



move_time = 1.5
hold_time = 0.05

# ------------------------

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# Register starting position
starting_pose = bus.sync_read("Present_Position")

# Move to zero config
move_to_pose_cubic(bus, zero_config, move_time)
hold_position(bus, 9999)

move_to_pose_cubic(bus, starting_pose, move_time)
bus.disable_torque()
