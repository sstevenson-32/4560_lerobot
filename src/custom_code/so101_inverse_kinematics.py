import numpy as np
import so101_forward_kinematics

def get_inverse_kinematics(target_position, target_orientation=None):
    "Geometric approach specific to the so-101 arms"

    # 1) Parse destination coordinates and combine to a single target matrix
    x_dest = target_position[0]
    y_dest = target_position[1]
    z_dest = target_position[2]
    # print(f"x_dest: {x_dest:.3f}, y_dest: {y_dest:.3f}, z_dest: {z_dest:.3f}")

    # 2) Solve for theta_1 (top view), what will get wrist directly above cube
    theta_1 = np.rad2deg( -np.atan( y_dest / (x_dest - 0.038835) ) )

    # 3) Get and parse target wrist position
    target_wrist_position = get_wrist_flex_position(target_position)
    x_wrist = target_wrist_position[0][0]
    y_wrist = target_wrist_position[0][1]
    z_wrist = target_wrist_position[0][2]

    g_w2 = so101_forward_kinematics.get_gw1(theta_1) @ so101_forward_kinematics.get_g12(0)
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
    theta_2 = np.rad2deg(np.pi/2 - (alpha + gamma) - delta) + 5
    theta_3 = np.rad2deg(np.pi/2 - beta + delta) + 5

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

    # 3) Parse target position
    x_target, y_target, z_target = float(target_position[0]), float(target_position[1]), float(target_position[2])

    g_w2 = so101_forward_kinematics.get_gw1(theta_1) @ so101_forward_kinematics.get_g12(0)
    g_w2_d = g_w2[0:3, 3]
    # print(f"x_offset: {g_w2_d[0]:.3f}, y_offset: {g_w2_d[1]:.3f}")
    # print(f"x_target: {(x_target - g_w2_d[0]):.3f}, y_target: {(y_target - g_w2_d[1]):.3f}")

    # Compute individual distance offsets and forward axis to consider direction
    dx = x_target - g_w2_d[0]
    dy = y_target - g_w2_d[1]
    fx = np.cos(theta_1_rad)
    fy = np.sin(theta_1_rad)

    # Compute signed distance along forward axis (positive in front, negative behind)
    dist_signed = dx * fx + dy * fy
    dist_target = np.sqrt(np.square(dx) + np.square(dy))
    z_target = z_target - g_w2_d[2]
    # print(f"dist_signed: {dist_signed:.6f}, dist_target: {dist_target:.6f}, z_target: {z_target:.6f}")

    l_1 = 0.11257
    l_2 = 0.1349 + 0.0611 + 0.1034  # Treat L2 as distance from {3} to {end effector}

    # 4) Solve for theta_2 and theta_3 (side view) using IK from lecture notes
    delta = 0.24378689318

    # Use r = sqrt(dist_signed^2 + z^2) when checking reachability
    r = np.sqrt(dist_signed**2 + z_target**2)
    reach = l_1 + l_2
    # eps = 1e-9
    # If outside reach, project onto the reachable boundary (preserve direction)
    if r > reach:
        # print(f"[IK] target outside reach (r={r:.6f}), projecting to r={reach:.6f}")
        scale = reach / r
        # scale radial (xy) and z components
        new_dist_signed = dist_signed * scale
        new_z = z_target * scale
        # update x,y/wrist so forward projection is preserved
        if dist_target > 1e-12:
            factor_xy = (new_dist_signed / dist_target) if dist_target != 0 else 0.0
            x_target = g_w2_d[0] + (dx) * factor_xy
            y_target = g_w2_d[1] + (dy) * factor_xy
            dist_target = np.sqrt((x_target - g_w2_d[0])**2 + (y_target - g_w2_d[1])**2)
        else:
            # purely along z-axis; keep x,y at base offset
            x_target = g_w2_d[0]
            y_target = g_w2_d[1]
            dist_target = 0.0
        z_target = g_w2_d[2] + new_z
        # recompute signed distance
        dx = x_target - g_w2_d[0]
        dy = y_target - g_w2_d[1]
        dist_signed = dx * fx + dy * fy
        r = np.sqrt(dist_signed**2 + (z_target - g_w2_d[2])**2)

    # compute safe cosine-law arguments and clip
    denom1 = 2.0 * l_1 * np.sqrt(dist_signed**2 + z_target**2) if (dist_signed**2 + z_target**2) > 1e-12 else 1e-12
    alpha_denom = (dist_signed**2 + z_target**2 + l_1**2 - l_2**2) / denom1
    alpha_denom_clipped = np.clip(alpha_denom, -1.0, 1.0)
    alpha = np.arccos(alpha_denom_clipped)

    denom2 = 2.0 * l_1 * l_2 if (2.0 * l_1 * l_2) > 1e-12 else 1e-12
    beta_denom = (l_1**2 + l_2**2 - dist_signed**2 - z_target**2) / denom2
    beta_denom_clipped = np.clip(beta_denom, -1.0, 1.0)
    beta = np.arccos(beta_denom_clipped)

    gamma = np.atan2(z_target, dist_signed)

    # print(f"alpha: {alpha}\n\talpha_denom: {alpha_denom}\nbeta: {beta}\n\tbeta_denom: {beta_denom}")

    # 5) Determine if we should use lefty or right orientation
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
    gwt = np.block([[so101_forward_kinematics.Ry(90), np.array(target_position).reshape(3,1)], [0, 0, 0, 1]])
    g4t = so101_forward_kinematics.get_g45(0) @ so101_forward_kinematics.get_g5t()

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

