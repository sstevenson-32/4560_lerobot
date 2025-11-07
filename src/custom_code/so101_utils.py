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
from so101_inverse_kinematics import get_inverse_kinematics

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