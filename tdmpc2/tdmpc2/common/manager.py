import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from .layers import mlp
from .math import lambda_returns


def soft_update(source, target, tau):
    """Soft update target network parameters."""
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class Manager(nn.Module):
    """
    Hierarchical Manager that selects discrete goals at low frequency.
    Uses policy gradient with value function baseline.
    """
    
    def __init__(self, z_dim, code_dim, hidden_dim=512, ent_coef=1e-3):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim
        self.ent_coef = ent_coef
        
        # Actor network: state -> goal code logits
        self.actor = mlp(z_dim, [hidden_dim], code_dim)
        
        # Value networks (online and target)
        self.value = mlp(z_dim, [hidden_dim], 1)
        self.target_value = mlp(z_dim, [hidden_dim], 1)
        
        # Initialize target network
        self.target_value.load_state_dict(self.value.state_dict())
        
        # Freeze target network
        for param in self.target_value.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def act(self, z, eval_mode=False):
        """
        Select a discrete goal code.
        
        Args:
            z: Current latent state [batch_size, z_dim]
            eval_mode: If True, use deterministic policy (argmax)
            
        Returns:
            code: One-hot goal code [batch_size, code_dim]
            log_prob: Log probability of selected code [batch_size]
            entropy: Policy entropy [batch_size]
        """
        logits = self.actor(z)
        dist = OneHotCategorical(logits=logits)
        
        if eval_mode:
            # Deterministic action (argmax)
            code = F.one_hot(logits.argmax(dim=-1), self.code_dim).float()
        else:
            # Stochastic action (sample)
            code = dist.sample()
        
        log_prob = dist.log_prob(code)
        entropy = dist.entropy()
        
        return code, log_prob, entropy
    
    def get_value(self, z, target=False):
        """
        Get value estimates for states.
        
        Args:
            z: Latent states [batch_size, z_dim]
            target: If True, use target network
            
        Returns:
            value: Value estimates [batch_size, 1]
        """
        if target:
            return self.target_value(z)
        else:
            return self.value(z)
    
    def update(self, batch, optimizer, gamma=0.99, lam=0.95, tau=0.005):
        """Update manager policy and value function."""
        z = batch['z']          # [seq_len, batch_size, z_dim]
        code = batch['code']    # [seq_len, batch_size, code_dim]
        reward = batch['reward']  # [seq_len, batch_size] or [seq_len-1, batch_size]
        done = batch['done']    # [seq_len, batch_size] or [seq_len-1, batch_size]
        
        seq_len, batch_size = z.shape[:2]
        
        # Flatten for network forward pass
        z_flat = z.view(-1, z.shape[-1])           # [seq_len * batch_size, z_dim]
        code_flat = code.view(-1, code.shape[-1])  # [seq_len * batch_size, code_dim]
        
        # Get value predictions for all states (including final state)
        value_pred = self.value(z_flat).squeeze(-1)  # [seq_len * batch_size]
        
        # For GAE, we need bootstrap value from target network
        with torch.no_grad():
            target_value_pred = self.target_value(z_flat).squeeze(-1)  # [seq_len * batch_size]
        
        # Handle rewards and done flags (ensure they have seq_len-1 length for transitions)
        if reward.shape[0] == seq_len:
            # If rewards include the final step, remove it
            reward = reward[:-1]  # [seq_len-1, batch_size]
        if done.shape[0] == seq_len:
            # If done flags include the final step, remove it
            done = done[:-1]      # [seq_len-1, batch_size]
            
        # Now we have seq_len-1 transitions and seq_len states (including bootstrap)
        assert reward.shape[0] == seq_len - 1, f"Reward shape mismatch: {reward.shape[0]} vs {seq_len-1}"
        assert done.shape[0] == seq_len - 1, f"Done shape mismatch: {done.shape[0]} vs {seq_len-1}"
        
        # Flatten transition data
        reward_flat = reward.view(-1)  # [seq_len-1 * batch_size]
        done_flat = done.view(-1)      # [seq_len-1 * batch_size]
        
        # Compute GAE for each trajectory in the batch
        returns = []
        advantages = []
        
        # Reshape data for trajectory processing
        value_seq = target_value_pred.view(seq_len, batch_size)  # [seq_len, batch_size]
        reward_seq = reward.view(seq_len-1, batch_size)          # [seq_len-1, batch_size]
        done_seq = done.view(seq_len-1, batch_size)              # [seq_len-1, batch_size]
        
        for b in range(batch_size):
            # Extract single trajectory
            v = value_seq[:, b]        # [seq_len] - includes bootstrap
            r = reward_seq[:, b]       # [seq_len-1] - transitions
            d = done_seq[:, b]         # [seq_len-1] - transitions
            
            # Compute returns using GAE (v has length seq_len, r has seq_len-1)
            ret = lambda_returns(r, v, 1.0 - d, gamma, lam)  # Returns [seq_len-1]
            adv = ret - v[:-1]  # Advantages using values for first seq_len-1 states
            
            returns.append(ret)
            advantages.append(adv)
        
        # Stack back to tensors
        returns = torch.stack(returns, dim=1).view(-1)     # [(seq_len-1) * batch_size]
        advantages = torch.stack(advantages, dim=1).view(-1)  # [(seq_len-1) * batch_size]
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
        # Get policy outputs for all states
        logits = self.actor(z_flat)  # [seq_len * batch_size, code_dim]
        dist = OneHotCategorical(logits=logits)
        log_prob_all = dist.log_prob(code_flat)  # [seq_len * batch_size]
        entropy_all = dist.entropy()             # [seq_len * batch_size]
        
        # Remove last timestep for loss computation (no corresponding return/advantage)
        log_prob = log_prob_all[:-batch_size]    # [(seq_len-1) * batch_size]
        entropy = entropy_all[:-batch_size]      # [(seq_len-1) * batch_size]
        value_pred_loss = value_pred[:-batch_size]  # [(seq_len-1) * batch_size]
        
        # Actor loss: policy gradient with entropy bonus
        actor_loss = -(log_prob * advantages.detach()).mean() - self.ent_coef * entropy.mean()
        
        # Critic loss: MSE with returns
        critic_loss = 0.5 * (value_pred_loss - returns.detach()).pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + critic_loss
        
        # Update parameters
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Soft update target network
        soft_update(self.value, self.target_value, tau)
        
        return {
            'manager/actor_loss': actor_loss.item(),
            'manager/critic_loss': critic_loss.item(),
            'manager/total_loss': total_loss.item(),
            'manager/entropy': entropy.mean().item(),
            'manager/value_mean': value_pred_loss.mean().item(),
            'manager/return_mean': returns.mean().item(),
            'manager/advantage_mean': advantages.mean().item(),
        } 