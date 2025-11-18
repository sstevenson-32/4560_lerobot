import time
import mujoco
import numpy as np
import math
from so101_mujoco_inverse_kinematics import get_end_effector_inverse_kinematics, get_inverse_kinematics
from so101_mujoco_forward_kinematics import get_forward_kinematics

def convert_to_dictionary(qpos):
    return {
        'shoulder_pan': qpos[0] * 180.0/3.14159, # convert to degrees
        'shoulder_lift': qpos[1] * 180.0/3.14159, # convert to degrees
        'elbow_flex': qpos[2] * 180.0/3.14159, # convert to degrees
        'wrist_flex': qpos[3] * 180.0/3.14159, # convert to degrees
        'wrist_roll': qpos[4] * 180.0/3.14159, # convert to degrees
        'gripper': qpos[5] * 100/3.14159 # convert to 0-100 range
    }

def convert_to_list(dictionary):
    return [
        dictionary['shoulder_pan'] * 3.14159/180.0,
        dictionary['shoulder_lift'] * 3.14159/180.0, 
        dictionary['elbow_flex'] * 3.14159/180.0, 
        dictionary['wrist_flex'] * 3.14159/180.0,
        dictionary['wrist_roll'] * 3.14159/180.0, 
        dictionary['gripper'] * 3.14159/100.0
    ]


def set_initial_pose(d, position_dict):
    pos = convert_to_list(position_dict)
    d.qpos = pos

# Send Position Commands to the Model
def send_position_command(d, position_dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos

def move_to_pose(m, d, viewer, desired_position, duration):
    start_time = time.time()
    starting_pose = d.qpos.copy()
    starting_pose = convert_to_dictionary(starting_pose)
    
    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # Interpolate each joint
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - alpha) * p0 + alpha * pf

        # Send command
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)
        
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

def move_to_pose_cubic(m, d, viewer, _, desired_position, duration):
    start_time = time.time()
    starting_pose = d.qpos.copy()
    starting_pose = convert_to_dictionary(starting_pose)

    # Set cubic coefficients based on current position
    a0 = starting_pose
    a1 = {joint: 0 for joint in starting_pose}
    a2 = {joint: ((3/math.pow(duration, 2)) * (desired_position[joint] - starting_pose[joint])) for joint in starting_pose}
    a3 = {joint: ((2/math.pow(duration, 3)) * (starting_pose[joint] - desired_position[joint])) for joint in starting_pose}

    # Create internal cubic interpolation helper
    def cubic_interpolation(t, joint):
        tlim = min(max(t, 0), duration)
        pos = a0[joint] + a1[joint]*tlim + a2[joint]*(tlim**2) + a3[joint]*(tlim**3)
        return pos
    
    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # Interpolate each joint
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = cubic_interpolation(t, joint)

        # Send command
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)
        
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()


def hold_position(m, d, viewer, duration):
    current_pos = d.qpos.copy()
    current_pos_dict = convert_to_dictionary(current_pos)
    
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        send_position_command(d, current_pos_dict)
        mujoco.mj_step(m, d)
        viewer.sync()


def pick_up_block_cubic(m, d, viewer, block_position, move_to_duration):
    # Move above block with gripper open
    block_raised = block_position.copy()
    block_raised[2] += 0.05  # raise block height amount
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 50
    move_to_pose_cubic(m, d, viewer, None, block_configuration_raised, move_to_duration)
    
    # Move down to block with gripper open
    block_configuration = get_inverse_kinematics(block_position)
    block_configuration['gripper'] = 50
    move_to_pose_cubic(m, d, viewer, None, block_configuration, 1.0)
    
    # Close gripper
    block_configuration_closed = block_configuration.copy()
    block_configuration_closed['gripper'] = 5
    move_to_pose_cubic(m, d, viewer, None, block_configuration_closed, 1.0)

    # Lift up again
    block_configuration_raised['gripper'] = 5
    move_to_pose_cubic(m, d, viewer, None, block_configuration_raised, move_to_duration)

def place_block_cubic(m, d, viewer, target_position, move_to_duration):
    
    # Move above target with gripper closed
    block_raised = target_position.copy()
    block_raised[2] += 0.03  # raise 1 inch
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 5
    move_to_pose_cubic(m, d, viewer, None, block_configuration_raised, move_to_duration)
    
    # Move down to block
    block_configuration = get_inverse_kinematics(target_position)
    block_configuration['gripper'] = 5
    move_to_pose_cubic(m, d, viewer, None, block_configuration, 1.0)
    
    # Open gripper 
    block_configuration_open = block_configuration.copy()
    block_configuration_open['gripper'] = 50
    move_to_pose_cubic(m, d, viewer, None, block_configuration_open, 1.0)
    
    # Return to raised position
    block_configuration_raised['gripper'] = 50
    move_to_pose_cubic(m, d, viewer, None, block_configuration_raised, move_to_duration)


def throw_obj(m, d, viewer, throw_velocity, throwing_pose, end_pose, time_to_throw, time_to_stop):
    # Setup timing args
    start_time = time.time()
    starting_pose = d.qpos.copy()
    starting_pose = convert_to_dictionary(starting_pose)

    # Solve coefficients to get from p(0) to p(throw), with p_dot(0) = 0, p_dot(throw) = throw_velocity
    start_point, start_rot = get_forward_kinematics(starting_pose)
    throw_point, throw_rot = get_forward_kinematics(throwing_pose)
    throwing_coefficients = eval_coeff_quintic(start_point, throw_point, [0.0, 0.0, 0.0], throw_velocity, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], time_to_throw)
    print(f"\n==================================================")
    print(f"start_point: {start_point}\nthrow_point: {throw_point}\nthrow_coeff: {throwing_coefficients}")

    # Solve coefficients to get from p(throw) to p(final), with p_dot(throw) = throw_velocity, p_dot(final) = 0
    end_point, end_rot = get_forward_kinematics(end_pose)
    stopping_coefficients = eval_coeff_quintic(throw_point, end_point, throw_velocity, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], time_to_stop)
    print(f"end_point: {end_point}\nstop_coeff: {stopping_coefficients}")
    print(f"==================================================\n")

    if (False):
        start_throw = eval_poly_const_accel(throwing_coefficients, 0.0)
        print(f"Moving to start throw: {start_throw}")
        start_config = get_end_effector_inverse_kinematics(start_throw)
        move_to_pose_cubic(m, d, viewer, None, start_config, 1.0)
        hold_position(m, d, viewer, 2.0)

        release_point = eval_poly_const_accel(throwing_coefficients, time_to_throw)
        print(f"Moving to end throw: {release_point}")
        release_config = get_end_effector_inverse_kinematics(release_point)
        move_to_pose_cubic(m, d, viewer, None, release_config, 1.0)
        hold_position(m, d, viewer, 2.0)

        return

    # Move to these positions
    positions = []
    times = []
    while True:
        t = time.time() - start_time
        if t > time_to_throw + time_to_stop:
            break

        # Use different coefficients as needed
        if t >= time_to_throw:
            # Use stopping coefficients
            target_point = eval_poly_quintic(stopping_coefficients, t - time_to_throw)
        else:
            # Use throwing coefficients
            target_point = eval_poly_quintic(throwing_coefficients, t)

        positions.append(target_point)
        times.append(t)
        # print(f"{t}:\t{target_point}")

        # Using IK, get target joint pos
        positions_dict = get_end_effector_inverse_kinematics(target_point)

        # Open gripper if near time_to_throw
        if (t >= (time_to_throw / 1.0)):
            positions_dict['wrist_roll'] = 90.0
            positions_dict['gripper'] = 50.0
        else:
            positions_dict['wrist_roll'] = 90.0
            positions_dict['gripper'] = 0.0

        # Move to this point
        send_position_command(d, positions_dict)
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

    if (True):
        # Plot the trajectory
        import matplotlib.pyplot as plt
        
        positions_array = np.array(positions)
        times_array = np.array(times)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle('End Effector Trajectory During Throw', fontsize=14, fontweight='bold')
        
        # Plot X position
        axes[0].plot(times_array, positions_array[:, 0], 'b-', linewidth=2)
        axes[0].axvline(x=time_to_throw, color='r', linestyle='--', label='Release Point')
        axes[0].set_ylabel('X Position (m)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot Y position
        axes[1].plot(times_array, positions_array[:, 1], 'g-', linewidth=2)
        axes[1].axvline(x=time_to_throw, color='r', linestyle='--', label='Release Point')
        axes[1].set_ylabel('Y Position (m)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot Z position
        axes[2].plot(times_array, positions_array[:, 2], 'm-', linewidth=2)
        axes[2].axvline(x=time_to_throw, color='r', linestyle='--', label='Release Point')
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Z Position (m)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()


# DEPRECATED: Solve coefficients and time based on initial and final config and velocities
def eval_coeff(start_point, end_point, start_vel, end_vel, time_period):
    # convert inputs to numpy float arrays so operations produce float arrays
    sp = np.array(start_point, dtype=float)
    ep = np.array(end_point, dtype=float)
    sv = np.array(start_vel, dtype=float)
    ev = np.array(end_vel, dtype=float)

    a0 = sp
    a1 = sv
    a2 = (3.0 / (time_period**2)) * (ep - sp) - (2.0 / time_period) * sv - (1.0 / time_period) * ev
    a3 = (2.0 / (time_period**3)) * (sp - ep) + (1.0 / (time_period**2)) * (sv + ev)

    return [a0, a1, a2, a3]

def eval_poly(coefficients, t):
    a0 = coefficients[0]
    a1 = coefficients[1]
    a2 = coefficients[2]
    a3 = coefficients[3]

    return a0 + a1*t + a2*t*t + a3*t*t*t



# Solve coefficients for constant acceleration trajectory (quadratic)
def eval_coeff_quintic(start_point, end_point, start_vel, end_vel, start_accel, end_accel, time_period):
    """
    Calculate coefficients for constant acceleration trajectory.
    Uses kinematic equations: p(t) = p0 + v0*t + 0.5*a*t^2
    """
    # convert inputs to numpy float arrays
    pos_i = np.array(start_point, dtype=float)
    pos_f = np.array(end_point, dtype=float)
    vel_i = np.array(start_vel, dtype=float)
    vel_f = np.array(end_vel, dtype=float)
    acc_i = np.array(start_accel, dtype=float)
    acc_f = np.array(end_accel, dtype=float)
    T = float(time_period)
    
    # 1. Calculate coefficients for t^0, t^1, t^2 (Start conditions)
    a0 = pos_i
    a1 = vel_i
    a2 = 0.5 * acc_i

    # 2. Solve for coefficients for t^3, t^4, t^5 (End conditions)
    # These are derived from the matrix inversion of the constraints at t=T
    
    # Common terms to simplify the calculation
    T2 = T**2
    T3 = T**3
    T4 = T**4
    T5 = T**5
    
    # Position delta
    h = pos_f - pos_i
    
    # These formulas come from solving the linear system Ax = B for the last 3 coeffs
    a3 = (1 / (2 * T3)) * (20 * h - (8 * vel_f + 12 * vel_i) * T - (3 * acc_f - acc_i) * T2)
    a4 = (1 / (2 * T4)) * (-30 * h + (14 * vel_f + 16 * vel_i) * T + (3 * acc_f - 2 * acc_i) * T2)
    a5 = (1 / (2 * T5)) * (12 * h - 6 * (vel_f + vel_i) * T - (acc_f - acc_i) * T2)

    return [a0, a1, a2, a3, a4, a5]

def eval_poly_quintic(coefficients, t):
    """Evaluate position for constant acceleration: p(t) = a0 + a1*t + a2*t^2"""
    a0 = coefficients[0]
    a1 = coefficients[1]
    a2 = coefficients[2]
    a3 = coefficients[3]
    a4 = coefficients[4]
    a5 = coefficients[5]
    
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t