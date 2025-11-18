import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position, pick_up_block_cubic, throw_obj
from so101_mujoco_inverse_kinematics import get_throw_theta_1, get_throwing_velocity, get_end_effector_inverse_kinematics
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
    # Configure numpy to print nicely
    np.set_printoptions(precision=3, suppress=True)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Specify object start and desired position
        start_obj_position = [0.2, 0.0, 0.0]
        # desired_obj_position = [0.4, 0.4, 0.0]
        desired_obj_position = [0.4, 0.0, 0.0]

        # Add a cylinder as a site for visualization
        show_cube(viewer, start_obj_position, np.eye(3))

        # 1) Pickup the target object
        # pick_up_block_cubic(m, d, viewer, start_obj_position, 1.0)


        # 2) Get to starting throwing position - Use theta to define rotation, all others are pre set
        theta_1 = get_throw_theta_1(desired_obj_position)
        
        starting_config = {
            'shoulder_pan': theta_1,
            'shoulder_lift': -60.0,
            'elbow_flex': -60.00,
            'wrist_flex': 0.0,
            'wrist_roll': 90.0,
            'gripper': 0
        }

        throw_config = {
            'shoulder_pan': theta_1,
            'shoulder_lift': -50.0,
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

        # print(f"========== Static Points ==========\n")
        # move_to_pose(m, d, viewer, starting_config, 1.0)
        # start_point, start_rot = get_forward_kinematics(starting_config)
        # print(f"Starting point: {start_point}")
        # hold_position(m, d, viewer, 2.0)

        # move_to_pose(m, d, viewer, throw_config, 1.0)
        # throw_point, throw_rot = get_forward_kinematics(throw_config)
        # print(f"Throw point: {throw_point}")
        # hold_position(m, d, viewer, 2.0)

        # move_to_pose(m, d, viewer, end_config, 1.0)
        # end_point, end_rot = get_forward_kinematics(end_config)
        # print(f"End point: {end_point}")
        # hold_position(m, d, viewer, 2.0)

        # print(f"\n==================================================")

        # Test new IK
        if (False):
            # START POINT
            joint_config = get_end_effector_inverse_kinematics([-0.153, 0, 0.467])
            move_to_pose(m, d, viewer, joint_config, 1.0)
            print(f"TEST: At starting point")
            hold_position(m, d, viewer, 1.0)

            # THROW POINT
            joint_config = get_end_effector_inverse_kinematics([0.009, 0, 0.515])
            move_to_pose(m, d, viewer, joint_config, 1.0)
            print(f"TEST: At release point")
            hold_position(m, d, viewer, 1.0)

            # END POINT
            joint_config = get_end_effector_inverse_kinematics([.221191747, 0,  .427755268])
            move_to_pose(m, d, viewer, joint_config, 2.0)
            print(f"TEST: At end point")
            hold_position(m, d, viewer, 1.0)

        move_to_pose(m, d, viewer, starting_config, 1.0)    # Reset to starting pos
        hold_position(m, d, viewer, 2.0)

        # Show a cube where the target is
        show_cube(viewer, desired_obj_position, np.eye(3))

        # 3) Get velocity to throw the ball at
        throw_velocity = get_throwing_velocity(theta_1, starting_config, throw_config, desired_obj_position)


        # 4) Throw the object
        time_to_throw = 2.0
        time_to_stop = 1.0
        throw_obj(m, d, viewer, theta_1, throw_velocity, throw_config, end_config, time_to_throw, time_to_stop)


        # Hold position for 10 seconds
        hold_position(m, d, viewer, 10.0)


# # Helper function to obtain random target position and yaw (rotation around the z-axis) from a given range 
# def get_random_position():
#     x_pos_range = [0.15, 0.25] #taken from workspace analysis
#     y_pos_range = [-0.2, 0.2] #taken from workspace analysis
#     yaw_range = [0, np.pi/2] #anything beyond 0 to 90 degrees is redundant due to symmetry of the cube
#     x = np.random.uniform(x_pos_range[0], x_pos_range[1])
#     y = np.random.uniform(y_pos_range[0], y_pos_range[1])
#     yaw = np.random.uniform(yaw_range[0], yaw_range[1])
#     return [x, y, 0.014], yaw

# # Start simulation with mujoco viewer
# def test_random_target():
#     with mujoco.viewer.launch_passive(m, d) as viewer:
#         # Pause for 10 seconds in order to make screen recording easier
#         #   time.sleep(10) 
#         while viewer.is_running():
#             for i in range(5):
#                 desired_position, desired_yaw = get_random_position()
#                 desired_orientation = np.eye(3)

#                 # Add a cylinder as a site for visualization
#                 show_cube(viewer, desired_position, desired_orientation)

#                 # target_wrist_position = get_wrist_flex_position(desired_position)
#                 # show_cube(viewer, target_wrist_position[0], np.eye(3))
                
#                 # Get the inverse kinematics solution
#                 joint_configuration = get_inverse_kinematics(desired_position, desired_orientation)

#                 # Move the robot to the desired pose
#                 move_to_pose(m, d, viewer, joint_configuration, 1.0)
                
#                 # Hold for two seconds for easy visualization
#                 hold_position(m, d, viewer, 1.0)

if __name__ == "__main__":
    test_basic()
    # test_random_target()