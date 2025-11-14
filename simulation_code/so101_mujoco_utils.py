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


def throw_obj(m, d, viewer, throw_velocity, throwing_pose, end_pose):
    # Setup timing args
    start_time = time.time()
    starting_pose = d.qpos.copy()
    starting_pose = convert_to_dictionary(starting_pose)

    # **** Tune these as needed ****
    time_to_throw = 5.0
    time_to_stop = 2.0
    # ******************************

    # Parse relevant data from starting_pose
    theta_1 = starting_pose['shoulder_pan']

    # Set constant throwing pose
    throwing_pose = {
        'shoulder_pan': theta_1,
        'shoulder_lift': -45.0,
        'elbow_flex': 0.00,
        'wrist_flex': 0.0,
        'wrist_roll': 90.0,
        'gripper': 0.0
    }

    # Solve coefficients to get from p(0) to p(throw), with p_dot(0) = 0, p_dot(throw) = throw_velocity
    start_point = get_forward_kinematics(starting_pose)[0]
    throw_point = get_forward_kinematics(throwing_pose)[0]
    throwing_coefficients = eval_coeff(start_point, throw_point, 0.0, throw_velocity, time_to_throw)
    print(f"start_point: {start_point}\nthrow_point: {throw_point}\nthrow_coeff: {throwing_coefficients}")

    # Solve coefficients to get from p(throw) to p(final), with p_dot(throw) = throw_velocity, p_dot(final) = 0
    end_point = get_forward_kinematics(end_pose)[0]
    stopping_coefficients = eval_coeff(throw_point, end_point, throw_velocity, 0.0, time_to_stop)
    print(f"end_point: {end_point}\nstop_coeff: {stopping_coefficients}")

    # Move to these positions
    while True:
        t = time.time() - start_time
        if t > 10:
            break

        # Get target point
        target_point = eval_poly(throwing_coefficients, t)

        # Using IK, get target joint pos
        positions_dict = get_end_effector_inverse_kinematics(target_point)

        # Move to this point
        send_position_command(d, positions_dict)
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

    # Using waypoints, move from p(0) to p(throw) and release

    # Using waypoints, move from p(throw) to p(final)


# Solve coefficients and time based on initial and final config and velocities
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