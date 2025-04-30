# control_tidybot_direct.py

import time
import numpy as np
from mujoco_env import MujocoEnv

def main():
    # Create a direct MujocoEnv
    env = MujocoEnv()

    # Reset the environment
    env.reset()

    # Define the action you want
    action = {
        'arm_pos': np.array([0.55, 0.0, 0.4]),  # Target XYZ
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
        'gripper_pos': np.array([0.5]),  # Gripper half open
        'base_pose': np.array([2.0, 0, 0.0]), 
    }



    i = 0
    # Keep the sim alive (optional)
    while i < 10000:
        
        # Directly step the environment
        env.step(action)
        time.sleep(0.1)
        i = i + 1

if __name__ == "__main__":
    main()
