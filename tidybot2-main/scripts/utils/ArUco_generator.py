import cv2
import cv2.aruco as aruco
import os
import numpy as np

# Create ArUco marker
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 27
marker_size = 500
marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size)

# Add white border (quiet zone) — by default 25% of marker size is a safe rule
border_percentage = 0.1
border_size = int(border_percentage * marker_size)
marker_with_border = cv2.copyMakeBorder(
    marker_image,
    top=border_size, bottom=border_size,
    left=border_size, right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=255  # white
)
# Show the marker
cv2.imshow("Marker with Border", marker_with_border)

# Save to disk
output_dir = os.path.join(os.path.dirname(__file__), "../../models/assets/aruco_markers")
os.makedirs(output_dir, exist_ok=True)
# Save the marker with border, specifying the file name and border size
marker_filename = f"marker{marker_id}.png"
save_path = os.path.join(output_dir, marker_filename)

print("Saving to:", os.path.abspath(save_path))
if cv2.imwrite(save_path, marker_with_border):
    print("✅ Marker with border saved successfully.")
else:
    print("❌ Failed to save marker.")
