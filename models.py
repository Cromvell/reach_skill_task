import torch
import torch.nn as nn
import copy

import gymnasium as gym
import common

class Encoder(nn.Module):
    def __init__(self, obs_shape, features_dim, num_layers=4, num_filters=32):
        super().__init__()

        if isinstance(obs_shape, gym.spaces.box.Box):
            obs_shape = obs_shape.shape
        self.obs_shape = obs_shape
        self.features_dim = features_dim
        self.num_layers = num_layers
        self.backbone = nn.Sequential(
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.ReLU()
        )
        for i in range(num_layers - 1):
            self.backbone.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            self.backbone.append(nn.ReLU())

        with torch.no_grad():
            x = torch.randn([32] + list(obs_shape))
            out_dim = self.backbone(x).shape[-1]

        self.outputs = {}

        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.features_dim)
        self.ln = nn.LayerNorm(self.features_dim)

    def forward(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.
        self.outputs['obs'] = obs

        if not isinstance(obs, torch.Tensor):
            obs = Encoder.obs_to_torch(obs)

        x = self.backbone(obs)
        self.outputs['backbone'] = x

        x = x.reshape(x.size(0), -1)

        x = self.fc(x)
        self.outputs['fc'] = x

        x = self.ln(x)
        self.outputs['ln'] = x

        return x

    @classmethod
    def obs_to_torch(self, obs):
        obs = torch.FloatTensor(obs)
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)
        return obs


class CURL(nn.Module):
    def __init__(self, obs_shape, z_dim, batch_size, encoder, encoder_target):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = encoder
        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def encode(self, x, ema=False) -> float:
        if ema:
            with torch.no_grad():
                z = self.encoder_target(x)
        else:
            z = self.encoder(x)

        return z

    def compute_logits(self, anchors: torch.tensor, augmented: torch.tensor):
        Wz = torch.matmul(self.W, augmented.T)
        logits = torch.matmul(anchors, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class RadAgent(object):
    def __init__(self, obs_shape, action_shape, encoder, encoder_target, device, args):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.encoder_tau = args.encoder_tau
        self.args = args


        self.encoder = encoder
        self.encoder_target = encoder_target

        self.augs = [ common.random_crop ]

        self.curl = CURL(
            obs_shape, args.encoder_features_dim, args.batch_size,
            self.encoder, self.encoder_target
        ).to(device)

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=args.encoder_lr
        )

        self.cpc_optimizer = torch.optim.Adam(
            self.curl.parameters(), lr=args.encoder_lr
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def augment(self, x):
        for aug in self.augs:
            x = aug(x)

        return x

    def update_cpc(self, obs_anchor, obs_positives, ema=False):
        z_anchor = self.curl.encode(obs_anchor)
        z_aug = self.curl.encode(obs_positives, ema=True)

        logits = self.curl.compute_logits(z_anchor, z_aug)
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        loss = self.cross_entropy_loss(logits, labels)

        # print(f"CONTRASTIVE LOSS: {loss}")

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()

        if ema:
            common.soft_update_params(
                self.encoder, self.encoder_target,
                self.encoder_tau
            )

    def preprocess_obs(self, obs_anchor, obs_positive):
        if obs_positive.shape[-1] != self.obs_shape:
            obs_positive = common.center_crop_images(obs_positive, self.obs_shape[-1])

        if obs_anchor.shape[-1] != self.obs_shape:
            obs_anchor = common.center_crop_images(obs_anchor, self.obs_shape[-1])

        obs_positive = torch.FloatTensor(obs_positive).to(self.device)
        obs_anchor = torch.FloatTensor(obs_anchor).to(self.device)

        return obs_anchor, obs_positive


    def learn(self, dataset, steps):
        for _ in range(steps):
            expert_obs, expert_actions = dataset.get_next_batch()

            obs_anchor = expert_obs.reshape([-1] + list(self.obs_shape))
            obs_positive = self.augment(obs_anchor)

            obs_anchor, obs_positive = self.preprocess_obs(obs_anchor, obs_positive)

            self.update_cpc(obs_anchor, obs_positive)

        del expert_obs, expert_actions
