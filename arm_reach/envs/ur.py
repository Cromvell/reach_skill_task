import os.path as osp
import math
import numpy as np

import pybullet as p
import pybullet_data


class UR_Arm:
    def __init__(self, urdf_dir: str, move_bounds: np.array):
        self.urdf_dir = urdf_dir
        self.enable_inverse_kinematics = True
        self.move_bounds = move_bounds
        self.max_force = 400
        self.max_velocity = 5.0
        self.end_effector_id = 7
        self.init_arm()

    def init_arm(self):
        def load_model(urdf_path, base_pos=None, base_orn=None, **args):
            obj_id = p.loadURDF(urdf_path, **args)

            if base_pos is not None and base_orn is None:
                p.resetBasePositionAndOrientation(obj_id, base_pos, [0, 0, 0, 1.0])
            if base_pos is not None and base_orn is not None:
                base_orn = p.getQuaternionFromEuler(base_orn)
                p.resetBasePositionAndOrientation(obj_id, base_pos, base_orn)

            return obj_id


        robot_flags = p.URDF_USE_SELF_COLLISION
        rel_urdf_path = osp.join(self.urdf_dir, "ur10_robot.urdf")
        self.robot_id = load_model(osp.join(osp.dirname(__file__), rel_urdf_path),
                                        base_pos=[0, 0, 0.62], base_orn=[0, 0, 0],
                                        useFixedBase=True, flags=robot_flags)

        self.joint_count = p.getNumJoints(self.robot_id)
        self.motor_names = []
        self.motor_ids = []

        for i in range(self.joint_count):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[3] > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_ids.append(int(joint_info[0]))

        self.joint_count = len(self.motor_ids)

        self.reset()

    def reset(self):

        # Moving arm to a random position
        lower_bound = max(self.move_bounds[2][0], 0.9)
        init_x = np.random.uniform(self.move_bounds[0][0], self.move_bounds[0][1])
        init_y = np.random.uniform(self.move_bounds[1][0], self.move_bounds[1][1])
        init_z = np.random.uniform(lower_bound, self.move_bounds[2][1])

        if self.enable_inverse_kinematics:
            init_ee_position = np.array([init_x, init_y, init_z])
            ik_init_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_id, init_ee_position)

            # Fix wrist orientation
            self.init_positions = list(ik_init_positions)
            self.init_positions[-1] = 0
            self.init_positions[-2] = math.pi/2.
            self.init_positions[-3] = 0

        else:
            # Right angle positions
            self.init_positions = [
                0.0, -math.pi/2., math.pi/2., -math.pi/2., 0.0, math.pi/2.
            ]

        for i in range(self.joint_count):
            joint_id = self.motor_ids[i]
            position = self.init_positions[i]
            p.resetJointState(self.robot_id, joint_id, position)
            p.setJointMotorControl2(self.robot_id,
                                    joint_id,
                                    p.POSITION_CONTROL,
                                    targetPosition=position,
                                    force=0,
                                    targetVelocity=0,
                                    positionGain=0,
                                    velocityGain=0)

    def act(self, command):
        if self.enable_inverse_kinematics:
            ee_pos = command

            joint_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_id, ee_pos, targetOrientation=[0, 0, 0])

            # Fix wrist orientation
            joint_positions = list(joint_positions)
            joint_positions[-1] = 0
            joint_positions[-2] = math.pi/2.
            joint_positions[-3] = 0

            for i in range(self.joint_count):
                joint_id = self.motor_ids[i]
                poisition = joint_positions[i]
                p.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                        jointIndex=joint_id,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_positions[i],
                                        targetVelocity=0,
                                        force=self.max_force,
                                        maxVelocity=self.max_velocity,
                                        positionGain=0.3,
                                        velocityGain=0.01)
        else:
            for joint_index in range(len(command)):
                motor_id = self.motor_ids[joint_index]
                p.setJointMotorControl2(self.robot_id,
                                        motor_id,
                                        p.POSITION_CONTROL,
                                        targetPosition=command[joint_index],
                                        force=self.max_force)

    def log(self):
        print(f"Model info: ")
        for joint_id in self.motor_ids:
            info = p.getJointInfo(self.robot_id, joint_id)
            print(f"{joint_id}: {info}")

    @property
    def real_positions(self):
        positions = []
        for i in range(self.joint_count):
            joint_id = self.motor_ids[i]
            state = p.getJointState(self.robot_id, joint_id)
            joint_position = state[0]
            positions.append(joint_position)
        return np.array(positions)


    @property
    def ee_real_position(self):
        ee_state = p.getLinkState(self.robot_id, self.end_effector_id)
        return np.array(ee_state[0])
