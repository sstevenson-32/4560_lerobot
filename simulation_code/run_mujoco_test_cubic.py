import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose_cubic, hold_position
from so101_mujoco_inverse_kinematics import get_inverse_kinematics
import numpy as np


# m = mujoco.MjModel.from_xml_path('simulation_code/model/scene_with_velocity.xml')
m = mujoco.MjModel.from_xml_path('simulation_code/model/scene.xml')
d = mujoco.MjData(m)

size_block = 0.0285 #in meters (1.125 inches)
pos1 = [0.25, 0.1, size_block/2]
pos2 = [0.25, -0.1, size_block/2]
config1 = get_inverse_kinematics(pos1)
config2 = get_inverse_kinematics(pos2)

# Helper Function to show a cube at a given position and orientation
def show_cube(viewer, position, orientation, geom_num=0, halfwidth=0.013):
    color = np.random.rand(3).tolist() + [1]  # Random RGB, alpha=0.2
    mujoco.mjv_initGeom(
      viewer.user_scn.geoms[geom_num],
      type=mujoco.mjtGeom.mjGEOM_BOX, 
      size=[halfwidth, halfwidth, halfwidth],                 
      pos=position,                         
      mat=orientation.flatten(),              
      rgba=color                           
    )
    viewer.user_scn.ngeom += 1
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
with mujoco.viewer.launch_passive(m, d) as viewer:
  
  # Show the target positions
  show_cube(viewer, pos1, np.eye(3), geom_num=0)
  show_cube(viewer, pos2, np.eye(3), geom_num=1)
  
  # Wait for one second before the simulation starts
  time.sleep(1.0) 
  
  # Cubic spline to first configuration
  move_to_pose_cubic(m, d, viewer, initial_config, config1, 2.0)
  hold_position(m, d, viewer, 1.0)

  # Cubic spline to second configuration
  move_to_pose_cubic(m, d, viewer, config1, config2, 2.0)
  hold_position(m, d, viewer, 1.0)

  # Cubic spline to initial configuration
  move_to_pose_cubic(m, d, viewer, config2, initial_config, 2.0)
  hold_position(m, d, viewer, 1.0)

