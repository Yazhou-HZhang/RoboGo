import time
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R
import cv2
from aprilTag import get_tag_pose_world


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mujoco_env import MujocoEnv
from constants import POLICY_CONTROL_PERIOD

def euler_to_wxyz(euler_angles):
    """
    Converts Euler angles (roll, pitch, yaw) in radians to quaternion in (w, x, y, z) format.

    Parameters:
        euler_angles (list or np.ndarray): Euler angles in radians [roll, pitch, yaw]

    Returns:
        np.ndarray: Quaternion as (w, x, y, z)
    """
    quat_xyzw = R.from_euler('xyz', euler_angles).as_quat()  # (x, y, z, w)
    quat_wxyz = np.roll(quat_xyzw, 1)  # convert to (w, x, y, z)
    return quat_wxyz

def compute_grasp_pose(object_pos, object_quat, offset_in_object=np.array([0, 0, 0.05])):
    """
    Computes the target gripper pose in the world frame given an object's pose and a local offset.
    Applies an additional 180° rotation around the Z-axis to the gripper orientation. (to keep camera up)

    Parameters:
        object_pos (np.ndarray): (3,) position of the object in the world frame.
        object_quat (np.ndarray): (4,) quaternion (xyzw) of the object in the world frame.
        offset_in_object (np.ndarray): (3,) desired offset in the object's local frame. Default is [0, 0, 0.05].

    Returns:
        gripper_pos_world (np.ndarray): (3,) desired arm position in world frame.
        gripper_quat_world (np.ndarray): (4,) desired arm orientation in world frame (same as object_quat).
    """
    # Convert object orientation to rotation
    r_obj = R.from_quat(object_quat)

    # Transform the offset from object frame to world frame
    offset_world = r_obj.apply(offset_in_object)

    # Compute target gripper position
    gripper_pos_world = object_pos + offset_world

    # Apply 180° rotation around Z-axis
    gripper_quat_world = (r_obj).as_quat()  # still xyzw
    
    # r_z180 = R.from_euler('z', np.pi)
    # gripper_quat_world = (r_obj * r_z180).as_quat()  # still xyzw

    return gripper_pos_world, gripper_quat_world

def move_to_target(env, target_pose, name="Target", tol_pos=1e-1, tol_quat=1e-1):
    def is_close(a, b, tol):
        return np.linalg.norm(a - b) < tol

    print(f"Moving to {name}...")
    while True:
        env.step(target_pose)
        time.sleep(POLICY_CONTROL_PERIOD)
        obs = env.get_obs()

        all_close = True
        for key in target_pose:
            if key == 'arm_quat':
                if not is_close(obs[key], target_pose[key], tol_quat):
                    all_close = False
                    break
            else:
                if not is_close(obs[key], target_pose[key], tol_pos):
                    all_close = False
                    break

        if all_close:
            break

    time.sleep(5)
    print(f"{name} reached.")




# World frame variable:
# keep fixed:
base_to_table_distance = 0.8
table_height = 0.3 # don't know why there is an offset in simulation, keep the -0.3 in z axis
end_effector_offset_1 = np.array([0.00, 0.0, -0.2]) # gripper open, forward to grasp position
end_effector_offset_2 = np.array([0.0, 0.0, -0])  # ready to grasp
q_delta = [0.5, 0.5, 0.5, -0.5] # apply rotation to container
base_tag_x = [0, 0, 0, 0, 0]       # tag locations for station i in y axis

# # xml setting
# water_container_pos = np.array([1.0, 0.0, 0.51 - table_height])
# water_container_euler = [-1.5708, -1.8, 0.0]
# q_delta = [0.5, 0.5, 0.5, -0.5] # apply rotation to container
# original_quat = euler_to_wxyz(water_container_euler)  # 我也不知道为啥这里要变 前面不用
# water_container_quat = (R.from_quat(q_delta) * R.from_quat(original_quat)).as_quat()
# rice_container_pos = np.array([1.0, -1.0, 0.5 - table_height])
# rice_container_quat = np.array([0.5, 0.5, 0.5, 0.5])

# CV, camera
fovy = 41.83792730009236
width, height = 640, 480
tag_size = 0.05 * 0.8 # scale by a factor of 0.8 (200mm -> 160mm)





def main():
    env = MujocoEnv(show_viewer=True, show_images=True)
    env.reset()
    time.sleep(2)
    obs = env.get_obs()         # dict_keys(['base_pose', 'arm_pos', 'arm_quat', 'gripper_pos', 'base_image', 'wrist_image'])


    # InspectionPose = { 
    #         'arm_pos': np.array([0.3, 0, 0.7]),
    #         'arm_quat': np.array([0.6123724, 0.6123724, 0.3535534, 0.3535534]),
    #         'gripper_pos': np.array([0.2]),
    #         'base_pose': np.array([0.0, 0.0, 0.0])
    #     }
    
    # move_to_target(env, InspectionPose, name="Inspection")

    # Pose to scan for tag
    target_pose_0 = {
        'base_pose': np.array([0.1, 0, 0]),
        'arm_pos': [0.6, 0, 0.31],
        'arm_quat': np.array([0.5792, 0.5792, 0.4056, 0.4056]),
        'gripper_pos': np.array([0.2]),
    }
    
    move_to_target(env, target_pose_0, name="scan for tag")
    
    
    # get tag location
    obs = env.get_obs() 

    # Call AprilTag detection to get water container pose
    get_image_func = lambda: env.get_obs()["wrist_image"]
    
    water_container_pos, water_container_quat = get_tag_pose_world(
        get_image_func=get_image_func,
        base_pose=obs["base_pose"],
        arm_pos=obs["arm_pos"],
        arm_quat=obs["arm_quat"],
        fovy=fovy,
        width=width,
        height=height,
        tag_size=tag_size
    )

    print(water_container_pos, water_container_quat)

    # First desired target
    base_pose_1 = np.array([(water_container_pos[0] - base_to_table_distance), base_tag_x[0], 0.0])
    gripper_pos_world_1, gripper_quat_world_1 = compute_grasp_pose(water_container_pos, water_container_quat, end_effector_offset_1)

    target_pose_1 = {
        'base_pose': base_pose_1,
        'arm_pos': gripper_pos_world_1 - base_pose_1,
        'arm_quat': gripper_quat_world_1,
        'gripper_pos': np.array([0.2]),
    }
    
    # print(target_pose_1['base_pose'])
    # print(target_pose_1['arm_pos'])
    # print(target_pose_1['arm_quat'])
    
    
    base_pose_2 = np.array([(water_container_pos[0] - base_to_table_distance), base_tag_x[0], 0.0])
    gripper_pos_world_2, gripper_quat_world_2 = compute_grasp_pose(water_container_pos, water_container_quat, end_effector_offset_2)
    
    target_pose_2 = {
        'base_pose': base_pose_2,
        'arm_pos': gripper_pos_world_2 - base_pose_2,
        'arm_quat': gripper_quat_world_2,
        'gripper_pos': np.array([0.2]),
    }
    # print(target_pose_2['base_pose'])
    # print(target_pose_2['arm_pos'])
    # print(target_pose_2['arm_quat'])
    
    
    target_pose_3 = {
        'gripper_pos': np.array([0.5])
    }
    
    target_pose_4 = {
        'arm_pos': target_pose_2['arm_pos'] + np.array([-0.3, 0.0, 0.2]),
        'arm_quat': np.array([0.5, 0.5, 0.5, 0.5])
    }
    
    target_pose_5 = {
        'arm_pos': target_pose_2['arm_pos'] + np.array([-0.3, 0.0, 0.2]),
        'arm_quat': np.array([0, 0.7071, 0.0, 0.7071])
    }
    
        
    move_to_target(env, target_pose_1, name="Target 1")
    move_to_target(env, target_pose_2, name="Target 2")
    # move_to_target(env, target_pose_3,tol_pos=1, name="Target 3")
    # move_to_target(env, target_pose_4, name="Target 4")
    # move_to_target(env, target_pose_5, name="Target 5")
    
    
    print("All targets reached!")
    
    
    # Keep simulation running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        
        # # Extract image (RGB) from observation
        # obs = env.get_obs()
        # image = obs["wrist_image"]  # shape: (H, W, 3), dtype: uint8

        # # Convert RGB to BGR for OpenCV
        # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # Save to disk
        # cv2.imwrite("scripts/wrist_view.png", image_bgr)
        # print("Saved camera image to scripts/wrist_view.png")
        
         
        print("Exiting simulation.")
        env.close()
    
    
    

if __name__ == "__main__":
    main()
