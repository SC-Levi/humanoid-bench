import torch
import numpy as np
from tensordict.tensordict import TensorDict
from .online_trainer import OnlineTrainer
from tdmpc2.common.goal_vae import GoalVAE
from tdmpc2.common.manager import Manager


class DirectorTrainer(OnlineTrainer):
    """
    Director trainer that implements hierarchical control with Manager and Worker.
    The Manager selects discrete goals at low frequency, while the Worker (original TD-MPC2)
    executes actions at high frequency guided by these goals.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # State tracking for hierarchical control
        self.goal_interval = self.cfg.manager.goal_interval
        self._current_goal = None
        self._current_code = None
        self._goal_step_counter = 0
        
        # Buffer for manager training
        self._manager_buffer = []
        
        self._setup_hierarchy()
        
    def _setup_hierarchy(self):
        """Initialize Goal-VAE and Manager components."""
        # Get latent dimension from the world model
        z_dim = self.cfg.latent_dim
        
        # Initialize Goal-VAE
        self.goal_vae = GoalVAE(
            z_dim=z_dim,
            n_latents=self.cfg.goal_vae.n_latents,
            n_classes=self.cfg.goal_vae.n_classes,
            tau_start=self.cfg.goal_vae.tau_start,
            tau_end=self.cfg.goal_vae.tau_end,
            anneal_steps=self.cfg.goal_vae.anneal_steps,
            beta=self.cfg.goal_vae.beta
        ).to(self.agent.device)
        
        # Initialize Manager
        self.manager = Manager(
            z_dim=z_dim,
            code_dim=self.goal_vae.code_dim,
            hidden_dim=self.cfg.manager.hidden_dim,
            ent_coef=self.cfg.manager.ent_coef
        ).to(self.agent.device)
        
        # Optimizers
        self.goal_vae_optimizer = torch.optim.Adam(
            self.goal_vae.parameters(), 
            lr=self.cfg.goal_vae.lr
        )
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(), 
            lr=self.cfg.manager.lr
        )
        
        print(f"Director setup complete:")
        print(f"  Goal-VAE: {z_dim} -> {self.goal_vae.code_dim} (discrete)")
        print(f"  Manager: {z_dim} -> {self.goal_vae.code_dim}")
        print(f"  Goal interval: {self.goal_interval}")
    
    def _get_goal(self, z, eval_mode=False):
        """Get goal from manager and decode it."""
        # Manager selects discrete code
        code, log_prob, entropy = self.manager.act(z, eval_mode=eval_mode)
        
        # Decode to continuous goal
        goal = self.goal_vae.decode(code)
        
        return goal, code, log_prob, entropy
    
    def _should_update_goal(self):
        """Check if we should update the goal."""
        return self._goal_step_counter % self.goal_interval == 0 or self._current_goal is None
    
    def _get_manager_reward(self, env_reward, z):
        """Compute manager reward (extrinsic + intrinsic)."""
        # Extrinsic reward (from environment)
        extrinsic_reward = env_reward
        
        # Intrinsic reward (Goal-VAE reconstruction error)
        with torch.no_grad():
            _, intrinsic_reward = self.goal_vae.kl_recon_loss(z)
            
        # Combined reward
        total_reward = extrinsic_reward + self.cfg.manager.intrinsic_coef * intrinsic_reward.item()
        
        return total_reward, intrinsic_reward.item()
    
    def eval(self):
        """Evaluate the hierarchical agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
            # Reset hierarchical state
            self._current_goal = None
            self._current_code = None
            self._goal_step_counter = 0
            
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
                
            while not done:
                # Get current latent state
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.agent.device)
                z = self.agent.model.encode(obs_tensor, None)
                
                # Update goal if needed
                if self._should_update_goal():
                    self._current_goal, self._current_code, _, _ = self._get_goal(z, eval_mode=True)
                
                # Worker (TD-MPC2) acts with goal conditioning
                action = self.agent.act(obs, t0=t == 0, eval_mode=True, goal=self._current_goal)
                
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                self._goal_step_counter += 1
                
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
                    
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            
            if self.cfg.save_video:
                self.logger.video.save(self._step, key='results/video')
                
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )
    
    def train(self):
        """Train the hierarchical agent."""
        train_metrics, done, eval_next = {}, True, True
        
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
                
            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False
                    
                if self._step > 0:
                    # Process manager trajectory for training
                    if len(self._manager_buffer) > 0:
                        self._process_manager_trajectory()
                    
                    train_metrics.update(
                        episode_reward=torch.tensor([td["reward"] for td in self._tds[1:]]).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())
                    
                    results_metrics = {
                        'return': train_metrics['episode_reward'],
                        'episode_length': len(self._tds[1:]),
                        'success': train_metrics['episode_success'],
                        'success_subtasks': info.get('success_subtasks', 0),
                        'step': self._step,
                    }
                    
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))
                
                # Reset for new episode
                obs = self.env.reset()[0]
                initial_td = self.to_td(obs)
                # Add dummy goal and code to maintain consistent keys
                initial_td['goal'] = torch.zeros(1, 512, dtype=torch.float32)  # Dummy goal vector with batch dim
                initial_td['code'] = torch.zeros(1, 64, dtype=torch.float32)   # Dummy code vector with batch dim
                self._tds = [initial_td]
                self._current_goal = None
                self._current_code = None
                self._goal_step_counter = 0
                self._manager_buffer = []
                
            # Collect experience
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.agent.device)
            z = self.agent.model.encode(obs_tensor, None)
            
            # Manager decision
            if self._should_update_goal():
                self._current_goal, self._current_code, log_prob, entropy = self._get_goal(z, eval_mode=False)
                
                # Store manager state for training
                manager_state = {
                    'z': z.detach().cpu(),
                    'code': self._current_code.detach().cpu(),
                    'log_prob': log_prob.detach().cpu(),
                    'entropy': entropy.detach().cpu(),
                    'step': self._goal_step_counter
                }
                self._manager_buffer.append(manager_state)
            
            # Worker action
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1, goal=self._current_goal)
            else:
                action = self.env.rand_act()
                
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            
            # Update manager buffer with reward
            if len(self._manager_buffer) > 0:
                manager_reward, intrinsic_reward = self._get_manager_reward(reward, z)
                self._manager_buffer[-1]['reward'] = manager_reward
                self._manager_buffer[-1]['intrinsic_reward'] = intrinsic_reward
            
            # Store transition
            td = self.to_td(obs, action, reward)
            # Always add goal and code fields for consistency
            if self._current_goal is not None:
                td['goal'] = self._current_goal.detach().cpu()
                td['code'] = self._current_code.detach().cpu()
            else:
                # Add dummy goal and code to maintain consistent keys
                td['goal'] = torch.zeros(1, 512, dtype=torch.float32)  # Dummy goal vector with batch dim
                td['code'] = torch.zeros(1, 64, dtype=torch.float32)   # Dummy code vector with batch dim
            self._tds.append(td)
            
            self._goal_step_counter += 1
            
            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                    
                for _ in range(num_updates):
                    _train_metrics = self._update_hierarchical()
                train_metrics.update(_train_metrics)
                
            self._step += 1
            
        self.logger.finish(self.agent)
    
    def _process_manager_trajectory(self):
        """Process completed manager trajectory and update manager."""
        if len(self._manager_buffer) < 2:  # Need at least 2 states for returns
            return
            
        # Accumulate rewards over goal intervals
        manager_rewards = []
        current_reward = 0
        
        for i, state in enumerate(self._manager_buffer):
            current_reward += state.get('reward', 0)
            
            # At the end of goal interval or trajectory
            if i == len(self._manager_buffer) - 1 or \
               (i + 1 < len(self._manager_buffer) and 
                self._manager_buffer[i + 1]['step'] % self.goal_interval == 0):
                manager_rewards.append(current_reward)
                current_reward = 0
        
        # Create manager batch
        if len(manager_rewards) > 1:
            batch = {
                'z': torch.stack([state['z'] for state in self._manager_buffer[:-1]]).to(self.agent.device),
                'code': torch.stack([state['code'] for state in self._manager_buffer[:-1]]).to(self.agent.device),
                'reward': torch.tensor(manager_rewards[:-1], dtype=torch.float32).to(self.agent.device),
                'done': torch.zeros(len(manager_rewards) - 1).to(self.agent.device),  # No early termination
                'log_prob': torch.stack([state['log_prob'] for state in self._manager_buffer[:-1]]).to(self.agent.device),
                'entropy': torch.stack([state['entropy'] for state in self._manager_buffer[:-1]]).to(self.agent.device),
            }
            
            # Update manager
            manager_metrics = self.manager.update(
                batch, 
                self.manager_optimizer, 
                gamma=self.cfg.manager.gamma,
                lam=self.cfg.manager.lam,
                tau=self.cfg.manager.tau
            )
            
            # Ensure 'step' key exists by merging common metrics
            manager_metrics.update(self.common_metrics())
            
            # Log manager metrics
            self.logger.log(manager_metrics, "train")
    
    def _update_hierarchical(self):
        """Update all components of the hierarchical agent."""
        metrics = {}
        
        # Sample batch from buffer
        batch = self.buffer.sample()
        
        # 1. Update Goal-VAE
        if 'goal' in batch:
            # Extract latent states
            obs = batch[0]  # [seq_len + 1, batch_size, ...]
            z_batch = []
            for t in range(obs.shape[0]):
                z_t = self.agent.model.encode(obs[t], None)
                z_batch.append(z_t)
            z_batch = torch.stack(z_batch)  # [seq_len + 1, batch_size, z_dim]
            
            # Goal-VAE loss
            vae_loss, recon_loss = self.goal_vae.kl_recon_loss(z_batch.view(-1, z_batch.shape[-1]))
            
            self.goal_vae_optimizer.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.goal_vae.parameters(), max_norm=10.0)
            self.goal_vae_optimizer.step()
            
            metrics.update({
                'goal_vae/total_loss': vae_loss.item(),
                'goal_vae/recon_loss': recon_loss.item(),
                'goal_vae/temperature': self.goal_vae._get_temperature().item(),
            })
        
        # 2. Update Worker (original TD-MPC2 agent)
        worker_metrics = self.agent.update(self.buffer)
        metrics.update({f'worker/{k}': v for k, v in worker_metrics.items()})
        
        return metrics 