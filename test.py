import gymnasium as gym
import arm_reach
import argparse

from stable_baselines3 import PPO
from models import CustomCombinedExtractor

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", default="weights/trained_policy")

    args = parser.parse_args()
    return args

config = parse_args()

config.policy_type = "MultiInputPolicy"
config.env_id = "ArmReach-v0"
config.total_timesteps = 10_000

config.encoder_features_dim = 50
config.encoder_extra_features_dim = 16
config.hidden_dim = 4

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(
        features_dim = config.encoder_features_dim,
        extra_features_dim = config.encoder_extra_features_dim,
        num_layers = 4,
        num_filters = 32,
    ),
    net_arch=[config.hidden_dim] * 3
)

model = PPO(config.policy_type, config.env_id, policy_kwargs=policy_kwargs, verbose=1)


env = model.get_env()
model = PPO.load(config.weights, env=env)

obs = env.reset()

step_count = 1
while True:
    action, _states = model.predict(obs)
    observations, reward, terminated, info = env.step(action)

    env.render()
    if terminated and step_count % 500 == 0:
        observation, info = env.reset()

    step_count += 1

env.close()
