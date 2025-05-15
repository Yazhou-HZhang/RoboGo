import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time

def compute_camera_in_world(base_pose, arm_pos, arm_quat):
    """
    Computes the camera's transformation matrix in the world frame.

    Parameters:
        base_pose (np.ndarray): (3,) robot base position in world frame.
        arm_pos (np.ndarray): (3,) end effector position in base frame.
        arm_quat (np.ndarray): (4,) quaternion of end effector in base frame (xyzw).

    Returns:
        np.ndarray: (4,4) camera-to-world transformation matrix.
    """
    # From XML
    cam_offset_pos = np.array([0, -0.05639, -0.058475])
    cam_offset_quat = np.array([0, 0, 0, 1])  # Identity quaternion

    # T_base_to_ee
    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = R.from_quat(arm_quat).as_matrix()
    T_base_ee[:3, 3] = arm_pos

    # T_world_to_base
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = base_pose

    # T_ee_to_cam
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R.from_quat(cam_offset_quat).as_matrix()
    T_ee_cam[:3, 3] = cam_offset_pos

    # T_world_to_cam = T_world_to_base * T_base_to_ee * T_ee_to_cam
    T_world_cam = T_world_base @ T_base_ee @ T_ee_cam
    return T_world_cam

def get_tag_pose_world(base_pose, arm_pos, arm_quat, T_camera_tag):
    
    T_world_camera = compute_camera_in_world(base_pose, arm_pos, arm_quat)
    T_world_tag = T_world_camera @ T_camera_tag

    tag_pos_world = T_world_tag[:3, 3]
    tag_rot_world = T_world_tag[:3, :3]
    tag_quat_world = R.from_matrix(tag_rot_world).as_quat()  # [x, y, z, w]
    tag_quat_world = np.roll(tag_quat_world, 1)              # → [w, x, y, z]

    return tag_pos_world, tag_quat_world
