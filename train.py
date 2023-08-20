import gymnasium as gym
import arm_reach

import torch

import copy
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from gail.dataset import ExpertDataset

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback

from models import CustomCombinedExtractor
from models import RadAgent

import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb_log", default=False, action='store_true')
    parser.add_argument("--pretrain_dataset", default='human_demonstration.npz')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--buffer_size", default=100000, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)

    parser.add_argument("--total_timesteps", default=20_000, type=int)
    parser.add_argument("--warmup_cpc", default=1600, type=int)
    parser.add_argument("--n_envs", default=1, type=int)
    parser.add_argument("--continue_train_path", default='')

    args = parser.parse_args()
    return args

config = parse_args()


#################################
#
#   Environment configuration
#
# W&B config:
config.policy_type     = "MultiInputPolicy"
config.env_id          = "ArmReach-v0"

# Encoder config
config.encoder_features_dim = 50
config.encoder_extra_features_dim = 16
config.encoder_tau          = 0.005
config.encoder_lr           = 1e-3
config.hidden_dim           = 1024

# RL config
config.n_steps       = 2000#4000
config.ent_coef      = 0.001
config.vf_coef       = 0.5
config.gamma         = 0.95
config.clip_range    = 0.2
config.clip_range_vf = None
config.target_kl     = None

def main():
    if config.wandb_log:
        run = wandb.init(
            project="arm_reach_skill",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            # save_code=True,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.wandb_log: tensorboard_log = f"runs/{run.id}"
    else: tensorboard_log = None

    render_mode = 'human'
    if config.n_envs > 1:
        render_mode = 'rgb_array'

    env_kwargs = dict(render_mode = render_mode)
    vec_env = make_vec_env(config.env_id, n_envs=config.n_envs, env_kwargs=env_kwargs)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim = config.encoder_features_dim,
            extra_features_dim = config.encoder_extra_features_dim,
            num_layers = 4,
            num_filters = 32,
        ),
        net_arch=[config.hidden_dim] * 3,
        normalize_images = True
    )

    model = PPO(
        config.policy_type, vec_env, policy_kwargs=policy_kwargs,
        verbose=1, tensorboard_log=tensorboard_log, learning_rate = config.lr,
        n_steps = config.n_steps, ent_coef=config.ent_coef, normalize_advantage=True,
        vf_coef = config.vf_coef, gamma = config.gamma, clip_range = config.clip_range,
        clip_range_vf = config.clip_range_vf, target_kl = config.target_kl
    )
    env = model.get_env()

    dataset = ExpertDataset(
        expert_path=config.pretrain_dataset, traj_limitation=-1,
        batch_size=config.batch_size, sequential_preprocessing=True
    )
    dataset.init_dataloader(config.batch_size)

    # Initiazlize policy
    def weight_init(m):
        """Custom weight init for Conv2D and Linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
            assert m.weight.size(2) == m.weight.size(3)
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
            mid = m.weight.size(2) // 2
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

    # Initialize policy
    if config.continue_train_path:
        model.load(config.continue_train_path)
    else:
        model.policy.apply(weight_init)

    print("Performing encoder pretraining...")
    # Contrastive pretraining step, updating model inplace
    encoder = model.policy.features_extractor
    encoder_target = copy.deepcopy(model.policy.features_extractor)
    pretrain_agent = RadAgent(
        env.observation_space, env.action_space, encoder, encoder_target, device, config
    )
    pretrain_agent.learn(dataset, config.warmup_cpc)

    # Fill model replay buffer with expert trajectories before RL policy training
    dataset.copy_to_rollout_buffer(model.rollout_buffer)

    # Copy pretrained parameters to model extractor
    model.policy.features_extractor.load_state_dict(pretrain_agent.encoder_target.state_dict())

    # Freezing extractor
    # for p in model.policy.features_extractor.parameters():
    #     p.requires_grad = False

    print("Starting policy training...")
    learn_callbacks = []
    if config.wandb_log:
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=1
        )
        learn_callbacks.append(wandb_callback)


    class CustomCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.obs_shape = env.observation_space.shape

            self.extractor_state_dict = copy.deepcopy(model.policy.features_extractor.state_dict())

        def _on_rollout_end(self) -> None:
            self.extractor_state_dict = copy.deepcopy(model.policy.features_extractor.state_dict())

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            pretrain_agent.encoder.load_state_dict(self.extractor_state_dict)
            pretrain_agent.encoder_target.load_state_dict(copy.deepcopy(self.extractor_state_dict))
            pretrain_agent.learn(dataset, steps=100)

            model.policy.features_extractor.load_state_dict(pretrain_agent.encoder_target.state_dict())

    # learn_callbacks.append(CustomCallback(verbose=1))


    model.learn(total_timesteps = config.total_timesteps, callback=learn_callbacks)
    model.save("weights/trained_policy")

    if config.wandb_log: run.finish()

if __name__ == "__main__":
    main()
