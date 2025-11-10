import time
import mujoco
import numpy as np
import math
from so101_mujoco_inverse_kinematics import get_inverse_kinematics

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


def throw_obj(m, d, viewer, target_position, target_joint_vel):
    # At target_position, must be at target_joint_vel with gripper open
    
    # Add logic to set final_position to be past target_position
    pass