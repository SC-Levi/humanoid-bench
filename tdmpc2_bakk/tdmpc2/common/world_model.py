from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from tdmpc2.common import layers, math, init


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, : cfg.action_dims[i]] = 1.0
        self._encoder = layers.enc(cfg)
        if cfg.use_moe:
            # Dynamics: 输入 dim = latent + action + (task?)
            dyn_in = cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0)
            # Gate 维度：single-task 用 latent+action；multi-task 用 task_dim
            gate_dim = cfg.task_dim if cfg.multitask else (cfg.latent_dim + cfg.action_dim)
            
            # Get annealing parameters from config with defaults
            moe_params = {
                'tau_init': getattr(cfg, 'moe_tau_init', 1.8),
                'tau_min': getattr(cfg, 'moe_tau_min', 0.5),
                'tau_max': getattr(cfg, 'moe_tau_max', 2.0),
                'beta': getattr(cfg, 'moe_beta', 0.02),
                'lb_alpha': getattr(cfg, 'moe_lb_alpha', 3e-2),
                'total_steps': getattr(cfg, 'steps', 200_000),
                'freeze_frac': getattr(cfg, 'moe_freeze_frac', 0.05),
            }
            
            self._dynamics = layers.MoEBlock(
                cfg=cfg,
                in_dim=dyn_in,
                gate_dim=gate_dim,
                hidden_dims=[cfg.mlp_dim, cfg.mlp_dim],
                out_dim=cfg.latent_dim,
                n_experts=cfg.n_experts,
                use_orthogonal=cfg.use_orthogonal,
                act=layers.SimNorm(cfg),
                **moe_params
            )
            # Reward 同理
            rew_in = cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0)
            self._reward = layers.MoEBlock(
                cfg=cfg,
                in_dim=rew_in,
                gate_dim=gate_dim,
                hidden_dims=[cfg.mlp_dim, cfg.mlp_dim],
                out_dim=max(cfg.num_bins, 1),
                n_experts=cfg.n_experts,
                use_orthogonal=cfg.use_orthogonal,
                **moe_params
            )
        else:
            self._dynamics = layers.mlp(
                cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
            self._reward = layers.mlp(
                cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                2 * [cfg.mlp_dim],
                max(cfg.num_bins, 1),
            )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        if cfg.use_moe:
            # If using MoE, initialize the head weight of the MoEBlock to zero
            init.zero_([self._reward.head.weight, self._Qs.params["1"]["weight"]])
        else:
            # If using regular mlp, initialize the last layer weight to zero
            init.zero_([self._reward[-1].weight, self._Qs.params["1"]["weight"]])
        
        # Create target Q-networks without using deepcopy to avoid TensorDict issues
        target_modules = []
        for _ in range(cfg.num_q):
            target_modules.append(
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
            )
        self._target_Qs = layers.Ensemble(target_modules)
        
        # Copy weights from main Q-networks to target Q-networks
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.copy_(p.data)
        
        # Disable gradients for target Q-networks
        for p in self._target_Qs.parameters():
            p.requires_grad_(False)
            
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def get_moe_aux_loss(self):
        """
        Get accumulated auxiliary losses from MoE blocks for load balancing.
        Returns total auxiliary loss from all MoE blocks.
        """
        if not self.cfg.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if hasattr(self._dynamics, 'aux_loss'):
            aux_loss += self._dynamics.aux_loss
        if hasattr(self._reward, 'aux_loss'):
            aux_loss += self._reward.aux_loss
        return aux_loss

    def zero_moe_aux_loss(self):
        """
        Reset accumulated auxiliary losses in MoE blocks.
        Should be called after each training step.
        """
        if not self.cfg.use_moe:
            return
        
        if hasattr(self._dynamics, 'zero_aux_loss'):
            self._dynamics.zero_aux_loss()
        if hasattr(self._reward, 'zero_aux_loss'):
            self._reward.zero_aux_loss()

    def get_moe_stats(self):
        """
        Get MoE statistics for monitoring and debugging.
        Returns dict with temperature, entropy, and gate statistics.
        """
        if not self.cfg.use_moe:
            return {}
        
        stats = {}
        if hasattr(self._dynamics, 'tau'):
            stats['dynamics_tau'] = self._dynamics.tau
            stats['dynamics_step'] = self._dynamics.global_step.item()
            if hasattr(self._dynamics, '_last_entropy'):
                stats['dynamics_entropy'] = self._dynamics._last_entropy.item()
        
        if hasattr(self._reward, 'tau'):
            stats['reward_tau'] = self._reward.tau
            stats['reward_step'] = self._reward.global_step.item()
            if hasattr(self._reward, '_last_entropy'):
                stats['reward_entropy'] = self._reward._last_entropy.item()
        
        return stats

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.use_moe:
            # MoEBlock expects separate z, a, task_emb arguments
            task_emb = None
            if self.cfg.multitask:
                if isinstance(task, int):
                    task = torch.tensor([task], device=z.device)
                task_emb = self._task_emb(task.long())
                if z.ndim == 3:
                    task_emb = task_emb.unsqueeze(0).repeat(z.shape[0], 1, 1)
                elif task_emb.shape[0] == 1:
                    task_emb = task_emb.repeat(z.shape[0], 1)
            return self._dynamics(z, a, task_emb)
        else:
            # Regular mlp expects concatenated input
            if self.cfg.multitask:
                z = self.task_emb(z, task)
            z = torch.cat([z, a], dim=-1)
            return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.use_moe:
            # MoEBlock expects separate z, a, task_emb arguments
            task_emb = None
            if self.cfg.multitask:
                if isinstance(task, int):
                    task = torch.tensor([task], device=z.device)
                task_emb = self._task_emb(task.long())
                if z.ndim == 3:
                    task_emb = task_emb.unsqueeze(0).repeat(z.shape[0], 1, 1)
                elif task_emb.shape[0] == 1:
                    task_emb = task_emb.repeat(z.shape[0], 1)
            return self._reward(z, a, task_emb)
        else:
            # Regular mlp expects concatenated input
            if self.cfg.multitask:
                z = self.task_emb(z, task)
            z = torch.cat([z, a], dim=-1)
            return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        
        # Import TensorDict here to avoid circular imports
        from tensordict import TensorDict
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": torch.ones_like(log_prob),
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }, device=mean.device)
        return action, info

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        `detach` specifies whether to detach gradients from the Q-networks.
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)
        
        if detach:
            out = out.detach()

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2
