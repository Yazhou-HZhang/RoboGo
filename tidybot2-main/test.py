import time
import numpy as np
from mujoco_env import MujocoEnv

def main():
    env = MujocoEnv()
    env.reset()

    action1 = {
        'arm_pos': np.array([0.8, 0, 0.5]),
        'arm_quat': np.array([0, 0, 0, 1]),
        'gripper_pos': np.array([1.0]),
        'base_pose': np.array([-0.2, -0.2, 0.0]),
    }


    action2 = {
        'arm_pos': np.array([1, 0, 0.5]),
        'arm_quat': np.array([0.7071, 0, 0, 0.7071]),
        'gripper_pos': np.array([1.0]),
        'base_pose': np.array([-0.2, -0.2, 0.0]),
    }

    # Send command for 30 steps
    for i in range(30):
        env.step(action1)
        time.sleep(0.1)
        print(f"{i + 1}\n")


    for i in range(100):
        env.step(action2)
        time.sleep(0.1)
        print(f"{i + 1}\n")


    # Keep sim alive (optional) until user stops it
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")

    # Proper cleanup
    env.close()

if __name__ == "__main__":
    main()
