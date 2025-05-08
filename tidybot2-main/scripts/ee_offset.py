import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_grasp_pose(object_pos, object_quat, offset_in_object=np.array([0, 0, 0.2])):
    """
    Computes the target gripper pose in the world frame given an object's pose and a local offset.

    Parameters:
        object_pos (np.ndarray): (3,) position of the object in the world frame.
        object_quat (np.ndarray): (4,) quaternion (xyzw) of the object in the world frame.
        offset_in_object (np.ndarray): (3,) desired offset in the object's local frame. Default is [0, 0, 0.2].

    Returns:
        gripper_pos_world (np.ndarray): (3,) desired gripper position in world frame.
        gripper_quat_world (np.ndarray): (4,) desired gripper orientation in world frame (same as object_quat).
    """
    # Convert object orientation to rotation
    r_obj = R.from_quat(object_quat)

    # Transform the offset from object frame to world frame
    offset_world = r_obj.apply(offset_in_object)

    # Compute target gripper position
    gripper_pos_world = object_pos - offset_world

    # Keep orientation same as object (can be adjusted)
    gripper_quat_world = object_quat

    return gripper_pos_world, gripper_quat_world

object_pos = np.array([1.05, -0.1, 0.1])
object_quat = np.array([0.5, 0.5, 0.5, 0.5])  # xyzw format

gripper_pos_world, gripper_quat_world = compute_grasp_pose(object_pos, object_quat)

target_pose = {
    'base_pose': np.array([0.0, 0.0, 0.035]),
    'arm_pos': gripper_pos_world - np.array([0.0, 0.0, 0.035]),  # arm_pos wrt base
    'arm_quat': gripper_quat_world,
    'gripper_pos': np.array([1.0]),
}

print(gripper_pos_world)
print(gripper_quat_world)