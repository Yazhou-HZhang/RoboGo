from scipy.spatial.transform import Rotation as R
import numpy as np


##########################################################################
# # Conver Euler to Quaternion

# # Euler angles in radians: (roll, pitch, yaw)
# euler_angles = [-1.5708, -2.3562, 0.0]  # XYZ convention

# # Convert to quaternion (x, y, z, w)
# quat = R.from_euler('xyz', euler_angles).as_quat()
# print("Quaternion (x, y, z, w):", np.round(quat, 4))


##########################################################################
# Convert Quaternion to Euler

# Quaternion in (x, y, z, w) format
quat = [0.6684, 0.6022, 0.2267, 0.3731]

# Convert to Euler angles (in radians) using XYZ convention
euler_rad = R.from_quat(quat).as_euler('xyz')

# Optionally convert to degrees
euler_deg = np.degrees(euler_rad)

print("Euler angles (rad):", np.round(euler_rad, 4))
print("Euler angles (deg):", np.round(euler_deg, 2))

##########################################################################
# # Relative rotation

# # Initial and target Euler angles in radians (XYZ convention)
# euler_from = [-1.5708, -1.5708, 0.0]
# euler_to = [1.5708, 0.0, 1.5708]

# # Convert to rotation objects
# rot_from = R.from_euler('xyz', euler_from)
# rot_to = R.from_euler('xyz', euler_to)

# # Compute the relative rotation (rot_delta * rot_from = rot_to)
# rot_delta = rot_to * rot_from.inv()

# # Express the delta as Euler or quaternion
# delta_euler = rot_delta.as_euler('xyz')
# delta_quat = rot_delta.as_quat()  # (x, y, z, w)

# print("Relative Euler rotation:", np.round(delta_euler, 4))
# print("Relative quaternion:", np.round(delta_quat, 4))

##########################################################################
# #Apply Rotation (Euler)

# # Existing orientation
# base_quat = [0.5, 0.5, 0.5, 0.5]  # Identity rotation

# # Rotation to apply: 90° about Z
# delta_rpy_rad = np.radians([0, 20, 0])
# delta = R.from_euler('xyz', delta_rpy_rad)

# # Combine the rotations
# base = R.from_quat(base_quat)
# new_quat = (delta * base).as_quat()
# print(np.round(new_quat, 4))
