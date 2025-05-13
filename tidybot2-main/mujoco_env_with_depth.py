'''
Added Depth frame for the mujoco simulation
Added Aruco marker detection and pose estimation in camera frame

To use this code, import mujoco_env_with_depth instead of mujoco_env
'''

import math
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from threading import Thread
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from constants import POLICY_CONTROL_PERIOD
from ik_solver import IKSolver
import os
MARKER_IMAGE_LENGTH         = 0.02  # m (edge length)
MARKER_IMAGE_BORDER_PERCENT = 0.05  # 5% of image size
MARKER_LENGTH               = MARKER_IMAGE_LENGTH *(1-MARKER_IMAGE_BORDER_PERCENT)


class ShmState:
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 3 + 4 + 1 + 1)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
        self.initialized[:] = 0.0

    def close(self):
        self.shm.close()

class ShmImage:
    def __init__(self,
                 camera_name=None, width=None, height=None,
                 existing_instance=None, depth=False):

        #  CASE 1 – attach to an existing shared‑memory block
        if existing_instance is not None:
            # Keep the original metadata
            self.camera_name = existing_instance.camera_name
            self.depth = existing_instance.depth
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

            dtype = np.float32 if self.depth else np.uint8
            self.data = np.ndarray(arr.shape, dtype=dtype, buffer=self.shm.buf)
            return

        # CASE 2 – create a brand‑new shared‑memory block 
        self.depth = depth
        self.camera_name = camera_name if not depth else f"{camera_name}_depth"

        shape = (height, width) if depth else (height, width, 3)
        dtype = np.float32 if depth else np.uint8
        arr = np.empty(shape, dtype=dtype)
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        self.data = np.ndarray(arr.shape, dtype=dtype, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self):
        self.shm.close()

# Adapted from https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py
class Renderer:
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.shm_image = ShmImage(existing_instance=shm_image)
        self.depth = shm_image.depth
        self.height, self.width = shm_image.data.shape[:2] if not self.depth else shm_image.data.shape

        # Set up scene and context
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.rect = mujoco.MjrRect(0, 0, self.width, self.height)

        # GL context
        self.gl_context = mujoco.gl_context.GLContext(self.width, self.height)
        self.gl_context.make_current()

        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        # Depth map mode
        if self.depth:
            self.mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR

        # Set up camera
        self.camera_name = shm_image.camera_name.replace('_depth', '')
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA.value, self.camera_name)
        if self.camera_id == -1:
            raise ValueError(f"Camera '{self.camera_name}' not found in model.")

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = self.camera_id

    def render(self):
        self.gl_context.make_current()

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.scene_option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)

        if self.depth:
            # Allocate temporary float32 buffer
            tmp = np.empty_like(self.shm_image.data, dtype=np.float32)
            mujoco.mjr_readPixels(None, tmp, self.rect, self.mjr_context)

            # Convert OpenGL depth buffer to actual depth (meters)
            extent = self.model.stat.extent
            near = self.model.vis.map.znear * extent
            far = self.model.vis.map.zfar * extent

            c = -0.5 * (far + near) / (far - near) - 0.5
            d = -0.5 * (2 * far * near) / (far - near)

            z_ndc = tmp.astype(np.float64)
            z_meters = d / (z_ndc + c)
            self.shm_image.data[:] = np.flipud(z_meters.astype(np.float32))

        else:
            tmp = np.empty_like(self.shm_image.data, dtype=np.uint8)
            mujoco.mjr_readPixels(tmp, None, self.rect, self.mjr_context)
            self.shm_image.data[:] = np.flipud(tmp)

    def close(self):
        if self.gl_context:
            self.gl_context.free()
        if self.mjr_context:
            self.mjr_context.free()
        self.gl_context = None
        self.mjr_context = None

class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

    def reset(self):
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()
            if 'base_pose' in command:
                # Set target base qpos
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position

class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper

        # IK solver
        self.ik_solver = IKSolver(ee_offset=0.12)

        # OTG (online trajectory generation)
        num_dofs = 7
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        self.otg_res = None

    def reset(self):
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()

            if 'arm_pos' in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi  # Unwrapped joint angles

                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if 'gripper_pos' in command:
                # Set target gripper pos
                self.ctrl_gripper[:] = 255.0 * command['gripper_pos']  # fingers_actuator, ctrlrange [0, 255]

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position

class MujocoSim:
    def __init__(self, mjcf_path, command_queue, shm_state, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['cube']:
            if object_name in body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        # Cache references to array slices
        base_dofs = self.model.body('base_link').jntnum.item()
        arm_dofs = 7
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs:(base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs:(base_dofs + arm_dofs)]
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        self.qpos_cube = self.data.qpos[(base_dofs + arm_dofs + 8):(base_dofs + arm_dofs + 8 + 7)]  # 8 for gripper qpos, 7 for cube qpos

        # Controllers
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper, self.model.opt.timestep)

        # Shared memory state for observations
        self.shm_state = ShmState(existing_instance=shm_state)

        # Variables for calculating arm pos and quat
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body('gen3/base_link').pos[2]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Reset cube
        self.qpos_cube[:2] += np.random.uniform(-0.1, 0.1, 2)
        theta = np.random.uniform(-math.pi, math.pi)
        self.qpos_cube[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(self, *_):
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()
        if command == 'reset':
            self.reset()

        # Control callbacks
        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)

        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        # Update arm pos
        # self.shm_state.arm_pos[:] = self.site_xpos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # Base orientation inverse
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # Arm pos in local frame

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        # self.shm_state.arm_quat[:] = self.site_quat
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  # right_driver_joint, joint range [0, 0.8]

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)

class MujocoEnv:
    def __init__(self, render_images=True, show_viewer=True, show_images=False):
        self.mjcf_path = 'models/stanford_tidybot/scene.xml'
        # self.mjcf_path = 'models/stanford_tidybot/scene.xml'
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)

        # Shared memory for state observations
        self.shm_state = ShmState()

        # Shared memory for image observations
        if self.render_images:
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))
                if camera_name == "wrist":
                    self.shm_images.append(ShmImage(camera_name, width, height, depth=True))

        # Start physics loop
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            # Start visualizer loop
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def physics_loop(self):
        # Create sim
        sim = MujocoSim(self.mjcf_path, self.command_queue, self.shm_state, show_viewer=self.show_viewer)

        # Start render loop
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        # Launch sim
        sim.launch()  # Launch in same thread as creation to avoid segfault

    def render_loop(self, model, data):
        # Set up renderers
        renderers = [Renderer(model, data, shm_image) for shm_image in self.shm_images]

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
            render_time = time.time() - start_time
            if render_time > 0.1:  # 10 fps
                print(f'Warning: Offscreen rendering took {1000 * render_time:.1f} ms, try making the Mujoco viewer window smaller to speed up offscreen rendering')

    def visualizer_loop(self):
        # ───────── 1.  Re‑attach to shared‑memory images  ─────────────────────────
        shm_images = [ShmImage(existing_instance=img) for img in self.shm_images]

        # Wait until each camera has produced its first frame.
        for img in shm_images:
            while np.all(img.data == 0):
                time.sleep(0.01)

        # ───────── 2.  Fetch the specific cameras we care about  ─────────────────
        cam_rgb   = next(img for img in shm_images if img.camera_name == "wrist")
        cam_depth = next(img for img in shm_images if img.camera_name == "wrist_depth")
        cam_base  = next(img for img in shm_images if img.camera_name == "base")

        # ───────── 3.  Decide scale so all windows are proportional  ────────────
        SCALE = 1      # 1.0 = native, 0.5 = half‑size, etc.

        def scaled_size(img):
            h, w = img.data.shape[:2]
            return int(w * SCALE), int(h * SCALE)

        w_rgb,   h_rgb   = scaled_size(cam_rgb)
        w_depth, h_depth = scaled_size(cam_depth)
        w_base,  h_base  = scaled_size(cam_base)

        # ───────── 4.  Screen positions  ─────────────────────────────────────────
        GAP_X, GAP_Y = 8, 40           # spacing between windows
        TOP_X, TOP_Y = 60, 60          # top‑left corner of grid

        pos_rgb   = (TOP_X, TOP_Y)
        pos_depth = (TOP_X + w_rgb + GAP_X, TOP_Y)
        pos_base  = (TOP_X, TOP_Y + h_rgb + GAP_Y)

        # Create windows once
        def create(img, size, pos):
            cv.namedWindow(img.camera_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(img.camera_name, *size)
            cv.moveWindow(img.camera_name, *pos)

        #convert to grayscale
        wrist_gray = cv.cvtColor(cam_rgb.data, cv.COLOR_RGB2GRAY)

        corners, ids, _ = cv.aruco.detectMarkers(wrist_gray, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250))
        print(corners, ids)

        # draw the detected markers on the image
        cv.aruco.drawDetectedMarkers(wrist_gray, corners, ids)

        # Use known image size
        height, width = wrist_gray.shape[:2]

        # Assume 60° vertical FoV (adjust as needed)
        fovy_deg = 60
        fovy_rad = np.deg2rad(fovy_deg)

        # Compute focal length from vertical FoV
        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy  # Assume square pixels
        cx = width / 2
        cy = height / 2

        # Camera intrinsics matrix
        cam_mtx = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        print("Camera Matrix:\n", cam_mtx)

        # No distortion in simulation
        dist_coef = np.zeros(5)
        print("Distortion Coefficients:\n", dist_coef)

        # create(cam_rgb,   (w_rgb,  h_rgb),   pos_rgb)
        # create(cam_depth, (w_depth, h_depth), pos_depth)
        # create(cam_base,  (w_base, h_base),  pos_base)


        # ───────── 5.  Refresh loop  ─────────────────────────────────────────────
        while True:
            start_time = time.time()
            # Wrist RGB
            bgr = cv.cvtColor(cam_rgb.data, cv.COLOR_RGB2BGR)
            # cv.imshow(cam_rgb.camera_name, bgr)

            depth_vis = cv.normalize(cam_depth.data, None, 0, 255,
                                    cv.NORM_MINMAX).astype(np.uint8)
            # cv.imshow(cam_depth.camera_name, depth_vis)

            base_bgr = cv.cvtColor(cam_base.data, cv.COLOR_RGB2BGR)
            # cv.imshow(cam_base.camera_name, base_bgr)

            wrist_gray = cv.cvtColor(cam_rgb.data, cv.COLOR_RGB2GRAY)
            corners, ids, _ = cv.aruco.detectMarkers(wrist_gray, cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250))
            # print(corners, ids)
            rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, cam_mtx, dist_coef)
            # print(rvec, tvec)
            # print id 's pose
            if ids is not None:
                cv.aruco.drawDetectedMarkers(wrist_gray, corners, ids)
                for i in range(len(ids)):
                    # Print marker pose in camera frame
                    print("Time {:.1f} ms | Marker {}: rvec = {}, tvec = {}".format((time.time() - start_time)*100, ids[i], rvec[i], tvec[i]))            # draw the detected markers on the image
            cv.aruco.drawDetectedMarkers(wrist_gray, corners, ids)
            # draw the axis of the detected markers
            cv.imshow("Detected Markers", wrist_gray)
            # move the window to the top right corner
            cv.moveWindow("Detected Markers", pos_rgb[0] + w_rgb + GAP_X, pos_rgb[1])
           
            # Wait for key press
            if cv.waitKey(30) == 27:
                break

        cv.destroyAllWindows()
        for img in shm_images:
            img.close()


    def reset(self):
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')

        # Wait for state publishing to initialize
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        # Wait for image rendering to initialize (Note: Assumes all zeros is not a valid image)
        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                time.sleep(0.01)

    def get_obs(self):
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        obs = {
            'base_pose': self.shm_state.base_pose.copy(),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.shm_state.gripper_pos.copy(),
        }
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
        return obs

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.command_queue.put(action)

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()
        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()


'''
This is a test class wrapper to save images from the mujoco simulation.
Note: This is not a part of the main functionality of the code.
Yicheng-dev
'''
class ImageSavingEnv:
    def __init__(self, env, camera_name='wrist', folder='test_images'):
        self.env = env
        self.camera_name = camera_name
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.counter = 0

    def step(self, action):
        self.env.step(action)
        # self.save_image()

    def reset(self):
        return self.env.reset()

    def get_obs(self):
        return self.env.get_obs()

    def close(self):
        self.env.close()

    def save_image(self):
        # Access shm_images directly
        shm_image = next((img for img in self.env.shm_images if img.camera_name == self.camera_name), None)
        if shm_image is not None:
            img_rgb = shm_image.data.copy()
            img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
            cv.imwrite(f'{self.folder}/{self.camera_name}_{self.counter:02d}.png', img_bgr)
            self.counter += 1

    def __getattr__(self, name):
        return getattr(self.env, name)
    

    def save_rgb_and_depth(self, count=10, interval=10):
        """
        Save RGB and depth images every `interval` frames, up to `count` total.
        """
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
            self._saved_count = 0

        self._frame_counter += 1

        if self._saved_count >= count:
            return

        if self._frame_counter % interval != 0:
            return

        rgb_image = next((img for img in self.env.shm_images if img.camera_name == self.camera_name), None)
        depth_image = next((img for img in self.env.shm_images if img.camera_name == f"{self.camera_name}_depth"), None)

        if rgb_image is not None:
            img_rgb = rgb_image.data.copy()
            img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
            rgb_path = f'{self.folder}/{self.camera_name}_rgb_{self._saved_count:03d}.png'
            cv.imwrite(rgb_path, img_bgr)
            print(f"[INFO] Saved RGB to {rgb_path}")

        if depth_image is not None:
            img_depth = depth_image.data.copy()
            img_depth_norm = cv.normalize(img_depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            depth_path = f'{self.folder}/{self.camera_name}_depth_{self._saved_count:03d}.png'
            cv.imwrite(depth_path, img_depth_norm)
            print(f"[INFO] Saved Depth to {depth_path}")

        self._saved_count += 1


'''
This is a test function to save images from the mujoco simulation.
it saves initial specific number of images from specific camera.
Note: This is not a part of the main functionality of the code.
Yicheng-dev
'''
def save_images(env, camera_name ='wrist', num_images_to_save=10):
    saved_images = 0
    while saved_images < num_images_to_save:
        shm_image = next((img for img in env.shm_images if img.camera_name == camera_name), None)
        if shm_image is not None:
            img_rgb = shm_image.data.copy()
            img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
            filepath = f'test_images/random_action_{camera_name}_{saved_images:03d}.png'
            cv.imwrite(filepath, img_bgr)
            # print(f"[INFO] Saved image to {filepath}")
            saved_images += 1
    print(f"[INFO] Saved {saved_images} images from camera '{camera_name}'.")


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
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
        gripper_pos_world = object_pos - offset_world

        # Apply 180° rotation around Z-axis
        r_z180 = R.from_euler('z', np.pi)
        gripper_quat_world = (r_obj * r_z180).as_quat()  # still xyzw

        return gripper_pos_world, gripper_quat_world

    # World frame variable:
    # keep fixed:
    base_to_table_distance = 0.9
    table_height = 0.3 # don't know why there is an offset in simulation, keep the -0.3 in z axis
    end_effector_offset_1 = np.array([0.00, 0.0, 0.05]) # gripper open, forward to grasp position
    end_effector_offset_2 = np.array([0.0, 0.0, 0.0])  # ready to grasp
    q_delta = [0.5, 0.5, 0.5, -0.5] # apply rotation to container

    tag_pos = [0, 0, 0, 0, 0]       # tag locations for station i in y axis

    # from CV / xml setting
    water_container_pos = np.array([1.0, 0.0, 0.5 - table_height])
    water_container_euler = [-1.5708, -2.0, 0.0]
    original_quat = euler_to_wxyz(water_container_euler)  
    water_container_quat = (R.from_quat(q_delta) * R.from_quat(original_quat)).as_quat()

    rice_container_pos = np.array([1.0, -1.0, 0.5 - table_height])
    rice_container_quat = np.array([0.5, 0.5, 0.5, 0.5])

    base_pose_1 = np.array([(water_container_pos[0] - base_to_table_distance), tag_pos[0], 0.0])
    gripper_pos_world_1, gripper_quat_world_1 = compute_grasp_pose(water_container_pos, water_container_quat, end_effector_offset_1)

    # env = MujocoEnv()
    # env = MujocoEnv(show_images=True)
    # env = MujocoEnv(render_images=False)
    env = MujocoEnv(render_images=True, show_images=True)
    env = ImageSavingEnv(env, camera_name='wrist', folder='test_images')

    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    # 'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    # 'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    # 'arm_quat': np.random.rand(4),
                    # 'gripper_pos': np.random.rand(1),
                    'base_pose': base_pose_1,
                    'arm_pos': gripper_pos_world_1 - base_pose_1,
                    'arm_quat': gripper_quat_world_1,
                    'gripper_pos': np.array([0.2]),
                }
                env.step(action)
                obs = env.get_obs()
                # print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                
                # save_images(env, camera_name = 'wrist' , num_images_to_save=10) # TEST: save images from wrist camera
                env.save_rgb_and_depth(count=10, interval=10)

                
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise

                # # show images (shm_image are images)
                # if env.show_images:
                #     for i, shm_image in enumerate(env.shm_images):
                #         cv.imshow(shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR))
                #         cv.moveWindow(shm_image.camera_name, 640 * i, -100)
                #     cv.waitKey(1)

                # print(env.render_images)
    finally:
        env.close()
