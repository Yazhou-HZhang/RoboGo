import cv2
import cv2.aruco as aruco
import os

# Create ArUco marker
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 27 # use larger than 20, smaller than 250
marker_size = 500
marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size)

# Ensure output directory exists save to relative path
output_dir = os.path.join(os.path.dirname(__file__), "../models/assets/aruco_markers")
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

# Full save path
save_path = os.path.join(output_dir, f"marker{marker_id}.png")
print("Saving to:", os.path.abspath(save_path))

# Save and verify
if cv2.imwrite(save_path, marker_image):
    print("✅ Marker saved successfully.")
else:
    print("❌ Failed to save marker. Check directory path and permissions.")
