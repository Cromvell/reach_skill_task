import math
import time
import numpy as np

import os.path as osp

import gymnasium as gym
import pybullet as p
import pybullet_data

from typing import Dict, List, Tuple, Any, Optional

from arm_reach.envs.ur import UR_Arm

class ArmReachEnv(gym.Env):
    metadata={'render_modes': ['human', 'rgb_array'], 'video.frames_per_second': 25}

    def __init__(self, render_mode=None):
        self.step_count = 0
        self.cam_height = 100
        self.cam_width = 100
        self.dense_reward = False

        self.client = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.ee_x_bounds = [0.3, 0.8]
        self.ee_y_bounds = [-0.3, 0.3]
        self.ee_z_bounds = [0.85, 1.0]
        self.ee_bounds = np.stack([self.ee_x_bounds, self.ee_y_bounds, self.ee_z_bounds])

        self.action_space = gym.spaces.box.Box(
            low  = self.ee_bounds[:, 0],
            high = self.ee_bounds[:, 1]
        )
        self.acts_low, self.acts_high = self.action_space.low, self.action_space.high

        self.observation_space = gym.spaces.box.Box(
            low = 0, high = 255,
            shape=(3, self.cam_width, self.cam_height),
            dtype=np.uint8
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # self.ik_controls = [
        #     p.addUserDebugParameter("EE x position", self.ee_x_bounds[0], self.ee_x_bounds[1],  self.ee_x_bounds[0] + 0.1),
        #     p.addUserDebugParameter("EE y position", self.ee_y_bounds[0], self.ee_y_bounds[1],  0.0),
        #     p.addUserDebugParameter("EE z position", self.ee_z_bounds[0], self.ee_z_bounds[1],  self.ee_z_bounds[0] + 0.1),
        # ]

        # self.joint_controls = {
        #     #                          Name            |Lower Bound|Upper Bound|Start pose
        #     1: p.addUserDebugParameter("Shoulder pan",  -2*math.pi, 2*math.pi,  0.0       ),
        #     2: p.addUserDebugParameter("Shoulder lift", -2*math.pi, 2*math.pi, -math.pi/2.),
        #     3: p.addUserDebugParameter("Elbow",         -2*math.pi, 2*math.pi,  math.pi/2.),
        #     4: p.addUserDebugParameter("Wrist 1",       -2*math.pi, 2*math.pi, -math.pi/2.),
        #     5: p.addUserDebugParameter("Wrist 2",       -2*math.pi, 2*math.pi, -math.pi/2.),
        #     6: p.addUserDebugParameter("Wrist 3",       -2*math.pi, 2*math.pi,  0.0       ),
        # }

        # Camera params
        camera_pos = [2.0, 0.0, 1.2]

        rot_q = p.getQuaternionFromEuler([0, math.pi/2.5, 0])
        rot_mat = np.array(p.getMatrixFromQuaternion(rot_q)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [0.0, 0.0, -1.0])
        up_vec = np.array([0.0, 0.0, 1.0])
        self.cam_view_matrix = p.computeViewMatrix(camera_pos, camera_pos + camera_vec, up_vec)
        self.cam_proj_matrix = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.01, farVal=100)

        self.done = False

    def reset(self, seed: int = None, options: Dict[str, Any] = None):
        self.step_count = 0

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        p.resetDebugVisualizerCamera(2.5, 90, -60, [0, 0, 0])

        self.arm = UR_Arm("../data/ur_description/urdf/", self.ee_bounds)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf")
        p.resetBasePositionAndOrientation(self.table_id, [0.5, 0, 0], [0, 0, 0, 1.0])

        # Select position for target
        def generate_target_pos(x_bounds, y_bounds):
            x = np.random.uniform(x_bounds[0], x_bounds[1])
            y = np.random.uniform(y_bounds[0], y_bounds[1])
            z = 0.7
            return np.array([x, y, z], dtype=np.float32)

        self.target_pos = generate_target_pos(x_bounds=[0.4, 0.8],
                                              y_bounds=[-0.4, 0.4])
        self.target_radius = 0.065
        self.target_id = p.loadURDF("sphere2red.urdf", globalScaling=1.6*self.target_radius)
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos, [0, 0, 0, 1.0])

        # self.arm.log()

        # Disable collision between arm and the target
        for i in self.arm.motor_ids:
            p.setCollisionFilterPair(self.target_id, self.arm.robot_id, -1, i, 0)
        p.setCollisionFilterPair(self.target_id, self.arm.robot_id, -1, self.arm.end_effector_id, 0)

        p.stepSimulation()

        ob = self.observe()
        info = dict()
        return ob, info

    def step(self, action):
        self.step_count += 1

        command = 0.05 * np.clip(action, -1, +1) # Constrain motion
        # print(f"COMMAND: {command}")
        self.arm.act(command)

        p.stepSimulation()

        ob = self.observe()
        time.sleep(1./240.)

        self.done = self.is_terminated()
        is_trunc = self.is_truncated()
        reward = self.compute_reward()

        return ob, reward, self.done, is_trunc, dict()

    def render(self):
        if self.render_mode != "rgb_array": return np.empty(shape=(self.cam_width, self.cam_height))
        return self._observations

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        p.disconnect(self.client)

    def observe(self):

        cam_arr = p.getCameraImage(self.cam_width, self.cam_height, self.cam_view_matrix, self.cam_proj_matrix)[2]
        cam_arr = np.reshape(cam_arr, (self.cam_width, self.cam_height, 4))
        cam_arr = np.transpose(cam_arr, (2, 0, 1)) # Adjust channels
        cam_arr = cam_arr[:3] # Throw away depth channel

        self._observations = cam_arr
        return cam_arr


    def compute_reward(self):
        if self.dense_reward:
            return -np.linalg.norm(self.arm.ee_real_position - self.target_pos)
        else:
            if self.done: return 0
            else:         return -1

    def is_truncated(self):
        if self.step_count > 500: return True
        return False

    def is_terminated(self):
        if np.linalg.norm(self.arm.ee_real_position - self.target_pos) < self.target_radius:
            return True

        return False
