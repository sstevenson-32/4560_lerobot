import mujoco
import mujoco.viewer
from so101_mujoco_utils import move_to_pose, hold_position

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:

    # All positions given in degrees
    initial_position = {
        'shoulder_pan': 0,
        'shoulder_lift': -100,
        'elbow_flex': 90,
        'wrist_flex': 90,
        'wrist_roll': 0,
        'gripper': 0,
        }
    
    target_pose = {
        'shoulder_pan': 0,
        'shoulder_lift': 0,
        'elbow_flex': 0,
        'wrist_flex': 0,
        'wrist_roll': 0,
        'gripper': 0,
        }
    
    # Start buffer before moving
    hold_position(m, d, viewer, 15)

    # Go to desired position
    move_to_pose(m, d, viewer, initial_position, 2)

    # Hold Position
    hold_position(m, d, viewer, 2)

    # Return to starting
    move_to_pose(m, d, viewer, target_pose, 2)

    # End buffer before closing
    hold_position(m, d, viewer, 10)
