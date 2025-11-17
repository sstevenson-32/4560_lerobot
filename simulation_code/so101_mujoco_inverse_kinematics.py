import numpy as np
import so101_mujoco_forward_kinematics

def get_inverse_kinematics(target_position, target_orientation=None):
    "Geometric approach specific to the so-101 arms"

    # 1) Parse destination coordinates and combine to a single target matrix
    x_dest = target_position[0]
    y_dest = target_position[1]
    z_dest = target_position[2]
    # print(f"x_dest: {x_dest:.3f}, y_dest: {y_dest:.3f}, z_dest: {z_dest:.3f}")

    # 2) Solve for theta_1 (top view), what will get wrist directly above cube
    # use atan2 for robustness; keep both rad and deg forms
    theta_1_rad = -np.atan2(y_dest, (x_dest - 0.038835))
    theta_1 = np.rad2deg(theta_1_rad)

    # 3) Get and parse target wrist position
    wrist_pos, wrist_orient = get_wrist_flex_position(target_position)
    x_wrist, y_wrist, z_wrist = float(wrist_pos[0]), float(wrist_pos[1]), float(wrist_pos[2])

    g_w2 = so101_mujoco_forward_kinematics.get_gw1(theta_1) @ so101_mujoco_forward_kinematics.get_g12(0)
    g_w2_d = g_w2[0:3, 3]
    # print(f"x_offset: {g_w2_d[0]}, y_offset: {g_w2_d[1]}")
    # print(f"x_target: {(x_wrist - g_w2_d[0]):.3f}, y_target: {(y_wrist - g_w2_d[1]):.3f}")

    dist_target = np.sqrt( np.square(x_wrist - g_w2_d[0]) + np.square(y_wrist - g_w2_d[1]) )
    z_target = z_wrist - g_w2_d[2]
    # print(f"dist_target: {dist_target:.3f}, z_target: {z_target:.3f}")

    l_1 = 0.11257
    l_2 = 0.1349

    # 4) Solve for theta_2 and theta_3 (side view)
    delta = 0.24378689318
    alpha = np.acos( (np.square(dist_target) + np.square(z_target) + np.square(l_1) - np.square(l_2)) / (2 * l_1 * np.sqrt(np.square(dist_target) + np.square(z_target))) )
    beta = np.acos( (np.square(l_1) + np.square(l_2) - np.square(dist_target) - np.square(z_target)) / (2 * l_1 * l_2) )
    gamma = np.atan2(z_target, dist_target)

    # 5) Determine if we should use lefty or right orientation
    theta_2 = np.rad2deg(np.pi/2 - (alpha + gamma) - delta)
    theta_3 = np.rad2deg(np.pi/2 - beta + delta)

    # 5) Solve for theta_4
    theta_4 = 90 - (theta_2 + theta_3)

    # 6) Solve for theta_5
    theta_5 = -theta_1

    # print(f"theta_1: {theta_1:.2f}, theta_2: {theta_2:.2f}, theta_3: {theta_3:.2f}, theta_4: {theta_4:.2f}, theta_5: {theta_5:.2f}")

    # Initialize the joint configuration dictionary
    joint_config = {
        'shoulder_pan': theta_1,
        'shoulder_lift': theta_2,
        'elbow_flex': theta_3,
        'wrist_flex': theta_4,
        'wrist_roll': theta_5,
        'gripper': 0.0
    }

    return joint_config

def get_end_effector_inverse_kinematics(target_position, target_orientation=None):
    "Geometric approach specific to the so-101 arms"

    # 1) Parse destination coordinates and combine to a single target matrix
    x_dest, y_dest, z_dest = float(target_position[0]), float(target_position[1]), float(target_position[2])
    # print(f"x_dest: {x_dest:.3f}, y_dest: {y_dest:.3f}, z_dest: {z_dest:.3f}")

    # 2) Solve for theta_1 (top view), what will end effector to target position
    # use atan2 for robustness and keep both rad and deg forms
    theta_1_rad = -np.atan(y_dest / (x_dest - 0.038835))
    theta_1 = np.rad2deg(theta_1_rad)

    # 3) Compute offsets and {1} -> {e} distance
    g_w2 = so101_mujoco_forward_kinematics.get_gw1(theta_1) @ so101_mujoco_forward_kinematics.get_g12(0)
    g_w2_d = g_w2[0:3, 3]
    # print(f"x_offset: {g_w2_d[0]:.3f}, y_offset: {g_w2_d[1]:.3f}")
    # print(f"x_target: {(x_target - g_w2_d[0]):.3f}, y_target: {(y_target - g_w2_d[1]):.3f}")

    dist_target = np.sqrt( np.square(x_dest - g_w2_d[0]) + np.square(y_dest - g_w2_d[1]) )
    z_target = z_dest - g_w2_d[2]
    # print(f"dist_target: {dist_target:.6f}, z_target: {z_target:.6f}")

    l_1 = 0.11257
    l_2 = 0.1349 + 0.0611 + 0.1034  # Treat L2 as distance from {3} to {end effector}
    delta = 0.24378689318

    # Elbow up if x >= 0, elbow down if x < 0
    if (x_dest < 0):
        dist_target = dist_target + 0.001
        # print(f"========== Solving for alpha ==========")
        # numerator = (np.square(dist_target) + np.square(z_target) + np.square(l_1) - np.square(l_2))
        # denominator = (2 * l_1 * np.sqrt(np.square(dist_target) + np.square(z_target)))
        # result = numerator / denominator
        # print(f"\tNumerator: {numerator}\n\tDenominator: {denominator}\n\tResult: {result}")

        # print(f"========== Solving for beta ==========")
        # numerator = (np.square(l_1) + np.square(l_2) - np.square(dist_target) - np.square(z_target))
        # denominator = (2 * l_1 * l_2)
        # result = numerator / denominator
        # print(f"\tNumerator: {numerator}\n\tDenominator: {denominator}\n\tResult: {result}")

        # 4) Solve for theta_2 and theta_3 (side view) using IK from lecture notes
        alpha = np.acos( (np.square(dist_target) + np.square(z_target) + np.square(l_1) - np.square(l_2)) / (2 * l_1 * np.sqrt(np.square(dist_target) + np.square(z_target))) )
        beta = np.acos( (np.square(l_1) + np.square(l_2) - np.square(dist_target) - np.square(z_target)) / (2 * l_1 * l_2) )
        gamma = np.atan2(z_target, dist_target)

        # 5) Solve for theta_1, theta_2
        theta_2 = np.rad2deg(-np.pi/2 + (gamma - alpha) - delta)
        theta_3 = np.rad2deg(np.pi/2 - beta + delta)
    else:
        # 4) Solve for theta_2 and theta_3 (side view) using IK from lecture notes
        alpha = np.acos( (np.square(dist_target) + np.square(z_target) + np.square(l_1) - np.square(l_2)) / (2 * l_1 * np.sqrt(np.square(dist_target) + np.square(z_target))) )
        beta = np.acos( (np.square(l_1) + np.square(l_2) - np.square(dist_target) - np.square(z_target)) / (2 * l_1 * l_2) )
        gamma = np.atan2(z_target, dist_target)

        # 5) Solve for theta_1, theta_2
        theta_2 = np.rad2deg(np.pi/2 - (alpha + gamma) - delta)
        theta_3 = np.rad2deg(np.pi/2 - beta + delta)

    # 5) Solve for theta_4
    theta_4 = 0

    # 6) Solve for theta_5
    theta_5 = 0

    # print(f"theta_1: {theta_1:.2f}, theta_2: {theta_2:.2f}, theta_3: {theta_3:.2f}, theta_4: {theta_4:.2f}, theta_5: {theta_5:.2f}")

    # Initialize the joint configuration dictionary
    joint_config = {
        'shoulder_pan': theta_1,
        'shoulder_lift': theta_2,
        'elbow_flex': theta_3,
        'wrist_flex': theta_4,
        'wrist_roll': theta_5,
        'gripper': 0.0
    }

    return joint_config


# Given target location, solve for position of wrist frame
# Assuming we want wrist frame directly above target position
def get_wrist_flex_position(target_position):
    # Ensure configuration, rotated about y 90 degrees, is accounted for
    gwt = np.block([[so101_mujoco_forward_kinematics.Ry(90), np.array(target_position).reshape(3,1)], [0, 0, 0, 1]])
    g4t = so101_mujoco_forward_kinematics.get_g45(0) @ so101_mujoco_forward_kinematics.get_g5t()

    gw4 = gwt @ np.linalg.inv(g4t)
    wrist_flex_position = gw4[0:3, 3]
    wrist_flex_orientation = gw4[0:3, 0:3]

    return wrist_flex_position, wrist_flex_orientation

# Given a target position, solve for the initial throwing position
# Rotates theta_1, else goes to a set position
def get_throw_theta_1(target_position):
    # Solve for theta_1 with inverse kinematics logic
    x_dest = target_position[0]
    y_dest = target_position[1]
    theta_1 = np.rad2deg( -np.atan( y_dest / (x_dest - 0.038835) ) )

    return theta_1

# Throwing velocity, velocity in x, y, and z direction
def get_throwing_velocity(theta_1, starting_pose, throwing_pose, target_block_pos):
    return [0.1, 0.0, 0.0]

