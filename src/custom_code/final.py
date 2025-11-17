from so101_utils import load_calibration, move_to_pose_cubic, hold_position, setup_motors, pick_up_block_cubic, place_block_cubic, throw_obj
from so101_inverse_kinematics import get_throw_theta_1, get_throwing_velocity

# CONFIGURATION VARIABLES
PORT_ID = "COM6"
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

start_obj_position = [0.2, 0.0, 0.0]
# desired_obj_position = [0.4, 0.4, 0.0]
desired_obj_position = [0.4, 0.0, 0.0]

move_time = 1.5
hold_time = 1.0

# ------------------------

calibration = load_calibration(ROBOT_NAME)
bus = setup_motors(calibration, PORT_ID)

# Register starting position
starting_pose = bus.sync_read("Present_Position")

# Move to zero config
# move_to_pose_cubic(bus, zero_config, move_time)
# hold_position(bus, hold_time)

# 1) Pickup the target object from a set position
pick_up_block_cubic(bus, start_obj_position, move_time)
hold_position(bus, hold_time)

# 2) Get required positions
theta_1 = get_throw_theta_1(desired_obj_position)

starting_config = {
'shoulder_pan': theta_1,
'shoulder_lift': -55.0,
'elbow_flex': -60.00,
'wrist_flex': 0.0,
'wrist_roll': 90.0,
'gripper': 0
}

throw_config = {
'shoulder_pan': theta_1,
'shoulder_lift': -45.0,
'elbow_flex': -45.00,
'wrist_flex': 0.0,
'wrist_roll': 90.0,
'gripper': 20.0
}

end_config = {
'shoulder_pan': theta_1,
'shoulder_lift': -45.0,
'elbow_flex': 0.00,
'wrist_flex': 0.0,
'wrist_roll': 90.0,
'gripper': 50.0
}

# 3) Move to starting pose
move_to_pose_cubic(bus, starting_config, move_time)

# 4) Get velocity to throw the ball at
throw_velocity = get_throwing_velocity(theta_1, starting_config, throw_config, desired_obj_position)


# 5) Throw the object
time_to_throw = 1.5
time_to_stop = 1.0
throw_obj(bus, theta_1, throw_velocity, throw_config, end_config, time_to_throw, time_to_stop)

# End at starting config
move_to_pose_cubic(bus, starting_pose, move_time)
bus.disable_torque()