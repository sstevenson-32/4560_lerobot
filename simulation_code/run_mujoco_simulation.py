import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:

    initial_position = {
        'shoulder_pan': 0,
        'shoulder_lift': -1.75,
        'elbow_flex': 1.69,
        'wrist_flex': 1.66,
        'wrist_roll': 0,
        'gripper': 0,
        }

    # Go to desired position

    # Hold Position

    # Return to starting
