# so101-utils.py
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from pathlib import Path
import draccus
import time
import math
from so101_inverse_kinematics import get_end_effector_inverse_kinematics, get_inverse_kinematics
from so101_forward_kinematics import get_forward_kinematics
import numpy as np

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


def load_calibration(ROBOT_NAME) -> None:
    """
    Helper to load calibration data from the specified file.

    Args:
        fpath (Path | None): Optional path to the calibration file. Defaults to `self.calibration_fpath`.
    """
    fpath = Path(f'calibration_files/{ROBOT_NAME}.json')
    with open(fpath) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
        return calibration

def setup_motors(calibration, PORT_ID):
    norm_mode_body = MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
                port=PORT_ID,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                    "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                    "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                    "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                    "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=calibration,
            )
    bus.connect(True)

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32) 
    return bus

def move_to_pose(bus, desired_position, duration):
    start_time = time.time()
    starting_pose = bus.sync_read("Present_Position")
    
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
        bus.sync_write("Goal_Position", position_dict, normalize=True)

        # (Optional) Read back
        present_pos = bus.sync_read("Present_Position")
        print(present_pos)

        time.sleep(0.02)  # 50 Hz loop

def hold_position(bus, duration):
    current_pos = bus.sync_read("Present_Position")
    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        bus.sync_write("Goal_Position", current_pos, normalize=True)
        time.sleep(0.02)  # 50 Hz loop

def move_to_pose_cubic(bus, desired_position, duration):
    start_time = time.time()
    starting_pose = bus.sync_read("Present_Position")

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
        bus.sync_write("Goal_Position", position_dict, normalize=True)
        
        present_pos = bus.sync_read("Present_Position")
        print(present_pos)

        time.sleep(0.02)  # 50 Hz loop


def pick_up_block(bus, block_position, move_to_duration):
    # Move above block with gripper open
    block_raised = block_position.copy()
    block_raised[2] += 0.05  # raise block height amount
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 50
    move_to_pose(bus, block_configuration_raised, move_to_duration)
    
    # Move down to block with gripper open
    block_configuration = get_inverse_kinematics(block_position)
    block_configuration['gripper'] = 50
    move_to_pose(bus, block_configuration, 1.0)
    
    # Close gripper
    block_configuration_closed = block_configuration.copy()
    block_configuration_closed['gripper'] = 5
    move_to_pose(bus, block_configuration_closed, 1.0)

    # Lift up again
    block_configuration_raised['gripper'] = 5
    move_to_pose(bus, block_configuration_raised, 1.0)

    return bus

def pick_up_block_cubic(bus, block_position, move_to_duration):
    # Move above block with gripper open
    block_raised = block_position.copy()
    block_raised[2] += 0.05  # raise block height amount
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 50
    move_to_pose_cubic(bus, block_configuration_raised, move_to_duration)
    
    # Move down to block with gripper open
    block_configuration = get_inverse_kinematics(block_position)
    block_configuration['gripper'] = 50
    move_to_pose_cubic(bus, block_configuration, 1.0)
    
    # Close gripper
    block_configuration_closed = block_configuration.copy()
    block_configuration_closed['gripper'] = 5
    move_to_pose_cubic(bus, block_configuration_closed, 1.0)

    # Lift up again
    block_configuration_raised['gripper'] = 5
    move_to_pose_cubic(bus, block_configuration_raised, 1.0)

    return bus


def place_block(bus, target_position, move_to_duration):
    # Move above target with gripper closed
    block_raised = target_position.copy()
    block_raised[2] += 0.03  # raise 1 inch
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 5
    move_to_pose(bus, block_configuration_raised, move_to_duration)
    
    # Move down to block
    block_configuration = get_inverse_kinematics(target_position)
    block_configuration['gripper'] = 5
    move_to_pose(bus, block_configuration, 1.0)
    
    # Open gripper 
    block_configuration_open = block_configuration.copy()
    block_configuration_open['gripper'] = 50
    move_to_pose(bus, block_configuration_open, 1.0)
    
    # Return to raised position
    block_configuration_raised['gripper'] = 50
    move_to_pose(bus, block_configuration_raised, 1.0)

    return bus

def place_block_cubic(bus, target_position, move_to_duration):
    # Move above target with gripper closed
    block_raised = target_position.copy()
    block_raised[2] += 0.03  # raise 1 inch
    block_configuration_raised = get_inverse_kinematics(block_raised)
    block_configuration_raised['gripper'] = 5
    move_to_pose_cubic(bus, block_configuration_raised, move_to_duration)
    
    # Move down to block
    block_configuration = get_inverse_kinematics(target_position)
    block_configuration['gripper'] = 5
    move_to_pose_cubic(bus, block_configuration, 1.0)
    
    # Open gripper 
    block_configuration_open = block_configuration.copy()
    block_configuration_open['gripper'] = 50
    move_to_pose_cubic(bus, block_configuration_open, 1.0)
    
    # Return to raised position
    block_configuration_raised['gripper'] = 50
    move_to_pose_cubic(bus, block_configuration_raised, 1.0)

    return bus


def throw_obj(bus, theta_1, throw_velocity, throwing_pose, end_pose, time_to_throw, time_to_stop):
    # Setup timing args
    start_time = time.time()
    starting_pose = bus.sync_read("Present_Position")

    # Solve for trajectory coefficients from start to throw
    start_point, start_rot = get_forward_kinematics(starting_pose)
    throw_point, throw_rot = get_forward_kinematics(throwing_pose)
    throwing_coeffs = eval_coeff_const_accel(
        start_point, throw_point, 
        [0.0, 0.0, 0.0], throw_velocity, 
        time_to_throw
    )
    
    # Solve for trajectory coefficients from throw to stop
    end_point, end_rot = get_forward_kinematics(end_pose)
    stopping_coeffs = eval_coeff_const_accel(
        throw_point, end_point, 
        throw_velocity, [0.0, 0.0, 0.0], 
        time_to_stop
    )
    
    # Move through continuous trajectory
    while True:
        t = time.time() - start_time
        if t > time_to_throw + time_to_stop:
            break

        # Determine which phase and calculate target point continuously
        if t < time_to_throw:
            # Throwing phase - use throwing coefficients
            target_point = eval_poly_const_accel(throwing_coeffs, t)
        else:
            # Stopping phase - use stopping coefficients
            phase_time = t - time_to_throw
            target_point = eval_poly_const_accel(stopping_coeffs, phase_time)

        # Using IK, get target joint pos
        positions_dict = get_end_effector_inverse_kinematics(target_point)

        # Open gripper if near time_to_throw
        if (t > (time_to_throw / 1.0)):
            positions_dict['wrist_roll'] = 90.0
            positions_dict['gripper'] = 50.0
        else:
            positions_dict['wrist_roll'] = 90.0
            positions_dict['gripper'] = 0.0

        # Send command
        bus.sync_write("Goal_Position", positions_dict, normalize=True)

        time.sleep(0.02)  # 50 Hz loop


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

def eval_poly_derivative(coefficients, t):
    a1 = coefficients[1]
    a2 = coefficients[2]
    a3 = coefficients[3]
    
    return a1 + 2.0*a2*t + 3.0*a3*t*t

# Solve coefficients for minimum-jerk trajectory (quintic polynomial)
def eval_coeff_const_accel(start_point, end_point, start_vel, end_vel, time_period):
    """
    Calculate coefficients for smooth minimum-jerk trajectory using quintic polynomial.
    This ensures continuous position, velocity, AND acceleration with zero jerk at endpoints.
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    Boundary conditions:
    p(0) = start_point, p(T) = end_point
    v(0) = start_vel, v(T) = end_vel  
    a(0) = 0, a(T) = 0 (smooth start and stop)
    """
    # convert inputs to numpy float arrays
    sp = np.array(start_point, dtype=float)
    ep = np.array(end_point, dtype=float)
    sv = np.array(start_vel, dtype=float)
    ev = np.array(end_vel, dtype=float)
    T = time_period
    
    # Quintic polynomial coefficients (derived from boundary conditions)
    a0 = sp
    a1 = sv
    a2 = np.zeros_like(sp)  # Zero acceleration at start
    
    # Solve for a3, a4, a5 using remaining boundary conditions
    a3 = (20.0*ep - 20.0*sp - (8.0*ev + 12.0*sv)*T) / (2.0*T**3)
    a4 = (30.0*sp - 30.0*ep + (14.0*ev + 16.0*sv)*T) / (2.0*T**4)
    a5 = (12.0*ep - 12.0*sp - (6.0*ev + 6.0*sv)*T) / (2.0*T**5)
    
    return [a0, a1, a2, a3, a4, a5]

def eval_poly_const_accel(coefficients, t):
    """Evaluate position for quintic polynomial: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5"""
    a0, a1, a2, a3, a4, a5 = coefficients
    return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t

def eval_poly_derivative_const_accel(coefficients, t):
    """Evaluate velocity for quintic: v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4"""
    a0, a1, a2, a3, a4, a5 = coefficients
    return a1 + 2.0*a2*t + 3.0*a3*t*t + 4.0*a4*t*t*t + 5.0*a5*t*t*t*t

def generate_waypoints(start_point, end_point, start_vel, end_vel, time_period, num_waypoints):
    """Generate waypoints with constant acceleration trajectory"""
    # Get coefficients for constant acceleration polynomial
    coeffs = eval_coeff_const_accel(start_point, end_point, start_vel, end_vel, time_period)
    
    waypoints = []
    for i in range(num_waypoints):
        t = (i / (num_waypoints - 1)) * time_period if num_waypoints > 1 else 0
        
        # Position at time t
        pos = eval_poly_const_accel(coeffs, t)
        
        # Velocity at time t (derivative of position polynomial)
        vel = eval_poly_derivative_const_accel(coeffs, t)
        
        waypoints.append((t, pos, vel))
    
    return waypoints