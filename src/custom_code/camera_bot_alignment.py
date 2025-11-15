from so101_utils import load_calibration, move_to_pose_cubic, hold_position, setup_motors, pick_up_block_cubic, place_block_cubic


# CONFIGURATION VARIABLES
PORT_ID = "COM3"
ROBOT_NAME = "follower-1"

# --- Specified Parameters ---

zero_config = {
    'shoulder_pan': 143.6043956043956,
    'shoulder_lift': -97.49450549450549,
    'elbow_flex': 94.68131868131869,
    'wrist_flex': -112.26373626373626,
    'wrist_roll': 59.56043956043956,
    'gripper': 0
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