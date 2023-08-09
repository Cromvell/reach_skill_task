import gymnasium as gym
import arm_reach

from stable_baselines3 import PPO
from models import Encoder

encoder_feature_dim = 50
hidden_dim = 4

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 10_000,
    "env_id": "ArmReach-v0",
}

policy_kwargs = dict(
    features_extractor_class=Encoder,
    features_extractor_kwargs=dict(
        features_dim = encoder_feature_dim,
    ),
    net_arch=[hidden_dim, hidden_dim, hidden_dim]
)

model = PPO(config["policy_type"], config["env_id"], policy_kwargs=policy_kwargs, verbose=1)


env = model.get_env()
model = PPO.load("weights/trained_policy", env=env)

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
