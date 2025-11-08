import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101_mujoco_inverse_kinematics import get_inverse_kinematics, get_wrist_flex_position
import numpy as np

m = mujoco.MjModel.from_xml_path('simulation_code/model/scene.xml')
d = mujoco.MjData(m)

# Helper Function to show a cube at a given position and orientation
def show_cube(viewer, position, orientation, geom_num=0, halfwidth=0.013):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_num],
        type=mujoco.mjtGeom.mjGEOM_BOX, 
        size=[halfwidth, halfwidth, halfwidth],                 
        pos=position,                         
        mat=orientation.flatten(),              
        rgba=[1, 0, 0, 0.2]                           
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()
    return

# Initial joint configuration at start of simulation
initial_config = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.00,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0          
}
set_initial_pose(d, initial_config)
send_position_command(d, initial_config)

# Start simulation with mujoco viewer
def test_basic():
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Specify the desired position of the cube to be picked up
        desired_position = [0.2, 0.2, 0.014]

        # Add a cylinder as a site for visualization
        show_cube(viewer, desired_position, np.eye(3))

        # First send the robot to a higher position with the gripper open
        joint_configuration = get_inverse_kinematics(desired_position, viewer)
        move_to_pose(m, d, viewer, joint_configuration, 1.0)

        # Show a cube where the wrist should be
        target_wrist_position = get_wrist_flex_position(desired_position)
        show_cube(viewer, target_wrist_position[0], np.eye(3))

        # Hold position for 10 seconds
        # hold_position(m, d, viewer, joint_configuration, 10.0)
        hold_position(m, d, viewer, 10.0)


# Helper function to obtain random target position and yaw (rotation around the z-axis) from a given range 
def get_random_position():
    x_pos_range = [0.15, 0.25] #taken from workspace analysis
    y_pos_range = [-0.2, 0.2] #taken from workspace analysis
    yaw_range = [0, np.pi/2] #anything beyond 0 to 90 degrees is redundant due to symmetry of the cube
    x = np.random.uniform(x_pos_range[0], x_pos_range[1])
    y = np.random.uniform(y_pos_range[0], y_pos_range[1])
    yaw = np.random.uniform(yaw_range[0], yaw_range[1])
    return [x, y, 0.014], yaw

# Start simulation with mujoco viewer
def test_random_target():
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Pause for 10 seconds in order to make screen recording easier
        #   time.sleep(10) 
        while viewer.is_running():
            for i in range(5):
                desired_position, desired_yaw = get_random_position()
                desired_orientation = np.eye(3)

                # Add a cylinder as a site for visualization
                show_cube(viewer, desired_position, desired_orientation)

                # target_wrist_position = get_wrist_flex_position(desired_position)
                # show_cube(viewer, target_wrist_position[0], np.eye(3))
                
                # Get the inverse kinematics solution
                joint_configuration = get_inverse_kinematics(desired_position, desired_orientation)

                # Move the robot to the desired pose
                move_to_pose(m, d, viewer, joint_configuration, 1.0)
                
                # Hold for two seconds for easy visualization
                hold_position(m, d, viewer, 1.0)

if __name__ == "__main__":
    # test_basic()
    test_random_target()