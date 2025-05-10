import cv2
import numpy as np
from pupil_apriltags import Detector
import math
from scipy.spatial.transform import Rotation as R
import time


def compute_camera_intrinsics(fovy_deg, width, height):
    fovy_rad = math.radians(fovy_deg)
    fy = height / (2 * math.tan(fovy_rad / 2))
    fx = fy  # assume square pixels
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def detect_apriltags(get_image_func, fovy, width, height, tag_size):
    """
    Continuously detects AprilTags from a grayscale image until a tag is found.
    
    Parameters:
        get_image_func (function): A function that returns the latest image (np.ndarray).
        fovy (float): Vertical field of view.
        width, height (int): Image dimensions.
        tag_size (float): Physical size of tag in meters.
    
    Returns:
        np.ndarray: 4x4 transformation matrix of the first detected tag.
    """
    # Compute camera intrinsics once
    fx, fy, cx, cy = compute_camera_intrinsics(fovy, width, height)
    
    # Initialize AprilTag detector
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    print("[INFO] Starting AprilTag detection loop...")

    while True:
        img = get_image_func()

        if img is None:
            print("[WARNING] No image available.")
            time.sleep(0.1)
            continue

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tags = at_detector.detect(img, estimate_tag_pose=True,
                                  camera_params=[fx, fy, cx, cy],
                                  tag_size=tag_size)

        if tags:
            tag = tags[0]
            print(f"[DETECTED] Tag ID: {tag.tag_id}")

            pos = tag.pose_t.flatten()
            rot_mat = tag.pose_R

            # Homogeneous transformation matrix: T_camera_tag
            T_camera_tag = np.eye(4)
            T_camera_tag[:3, :3] = rot_mat
            T_camera_tag[:3, 3] = pos

            # Optional visualization
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            corners = np.int32(tag.corners)
            cv2.polylines(img_color, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img_color, f"ID: {tag.tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("AprilTag Detection", img_color)
            cv2.waitKey(1)
            return T_camera_tag

        print("[INFO] No tag detected. Retrying...")
        time.sleep(0.1)
    
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

def get_tag_pose_world(get_image_func, base_pose, arm_pos, arm_quat, fovy, width, height, tag_size):
    T_camera_tag = detect_apriltags(
        get_image_func=get_image_func,
        fovy=fovy,
        width=width,
        height=height,
        tag_size=tag_size
    )

    T_world_camera = compute_camera_in_world(base_pose, arm_pos, arm_quat)
    T_world_tag = T_world_camera @ T_camera_tag

    tag_pos_world = T_world_tag[:3, 3]
    tag_rot_world = T_world_tag[:3, :3]
    tag_quat_world = R.from_matrix(tag_rot_world).as_quat()  # [x, y, z, w]
    tag_quat_world = np.roll(tag_quat_world, 1)              # → [w, x, y, z]

    return tag_pos_world, tag_quat_world






# if __name__ == "__main__":
#     # Camera parameters
#     image_path = "scripts/wrist_view.png"
#     tag_size = 0.06  # meters
#     fovy = 41.83792730009236
#     width, height = 640, 480

#     # Robot observation (from env.get_obs())
#     base_pose = np.array([0.1, 0.0, 0.0])
#     arm_pos = np.array([0.6, 0.0, 0.31])
#     arm_quat = np.array([0.5792, 0.5792, 0.4056, 0.4056]) 

#     # 1. Detect AprilTag in camera frame
#     T_camera_tag = detect_apriltags(
#         img_path=image_path,
#         fovy=fovy,
#         width=width,
#         height=height,
#         tag_size=tag_size
#     )

#     # 2. Compute camera pose in world frame
#     T_world_camera = compute_camera_in_world(
#         base_pose=base_pose,
#         arm_pos=arm_pos,
#         arm_quat=arm_quat
#     )

#     # 3. Transform tag pose to world frame
#     T_world_tag = T_world_camera @ T_camera_tag
#     tag_pos_world = T_world_tag[:3, 3]
#     tag_rot_world = T_world_tag[:3, :3]
#     quat = R.from_matrix(tag_rot_world).as_quat()  # (x, y, z, w)
    
#     # Output result
#     print("\n[RESULT] Tag position in world frame:\n", np.round(tag_pos_world, 4))
#     print(f"[RESULT] Tag orientation (quaternion in world frame): x={quat[0]:.4f}, y={quat[1]:.4f}, z={quat[2]:.4f}, w={quat[3]:.4f}")


# expected output:
# position: [1.0, 0.0, 0.21]
# Orientation: [-0.55389496,  0.43954361, -0.43954564,  0.55389658], [-90.    -0.   -76.87]