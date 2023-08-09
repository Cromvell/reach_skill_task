import numpy as np
import argparse

import arm_reach
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from gail.record_expert import generate_expert_traj

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--dataset_name", default="human_demonstration")

    args = parser.parse_args()
    return args

config = parse_args()

config.env_id = "ArmReach-v0"

class HumanExpert():
    def __init__(self, env):
        self.env = env
        self.ee_last = None

    def __call__(self, _obs):
        if self.ee_last is None:
            self.ee_last = self.env.envs[0].arm.ee_real_position
            return np.zeros((1, 3), dtype=np.float32)

        ee_actual = self.env.envs[0].arm.ee_real_position

        ee_displacement = ee_actual - self.ee_last
        self.ee_last = ee_actual

        ee_displacement = ee_displacement.clip(min=1e-6)
        ee_displacement = ee_displacement[np.newaxis, ...]

        print(f"OBS: {_obs.shape}")
        print(f"DISP: {ee_displacement}")

        return ee_displacement

def main():
    env = DummyVecEnv([lambda: gym.make(config.env_id)])

    human_expert = HumanExpert(env)
    generate_expert_traj(human_expert, config.dataset_name, env, n_episodes=config.n_episodes)

    print(f"Dataset successfully saved into {config.dataset_name}.npz")

if __name__ == "__main__":
    main()
