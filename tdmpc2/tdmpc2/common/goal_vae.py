import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .layers import mlp


class GoalVAE(nn.Module):
    """
    Goal Variational Autoencoder for discrete goal representation.
    Encodes continuous latent states into discrete one-hot codes and back.
    """
    
    def __init__(self, z_dim, n_latents=8, n_classes=8, tau_start=1.0, tau_end=0.1, anneal_steps=1e5, beta=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.beta = beta
        
        # Encoder: z_dim -> n_latents * n_classes (logits for each categorical)
        self.encoder = mlp(z_dim, [2 * z_dim], n_latents * n_classes)
        
        # Decoder: n_latents * n_classes -> z_dim
        self.decoder = mlp(n_latents * n_classes, [2 * z_dim], z_dim)
        
        # Step counter for temperature annealing
        self.register_buffer('steps', torch.zeros(()))
        
    @property
    def code_dim(self):
        """Return the dimensionality of the discrete code space."""
        return self.n_latents * self.n_classes
    
    def _get_temperature(self):
        """Compute current temperature for Gumbel-Softmax annealing."""
        progress = torch.clamp(self.steps / self.anneal_steps, 0.0, 1.0)
        tau = self.tau_start * torch.exp(-progress * torch.log(torch.tensor(self.tau_start / self.tau_end)))
        return tau
    
    def encode(self, z):
        """
        Encode continuous latent states to discrete one-hot codes.
        
        Args:
            z: Continuous latent states [batch_size, z_dim]
            
        Returns:
            code: One-hot codes [batch_size, n_latents * n_classes]
            dist: Categorical distribution for KL computation
        """
        batch_size = z.size(0)
        
        # Get logits from encoder
        logits = self.encoder(z)  # [B, n_latents * n_classes]
        logits = logits.view(batch_size, self.n_latents, self.n_classes)
        
        # Create relaxed categorical distribution with current temperature
        tau = self._get_temperature()
        dist = RelaxedOneHotCategorical(tau, logits=logits)
        
        # Sample using straight-through estimator
        code = dist.rsample()  # [B, n_latents, n_classes]
        code = code.view(batch_size, -1)  # [B, n_latents * n_classes]
        
        return code, dist
    
    def encode_discrete(self, z, eval_mode=False):
        """
        Encode to discrete one-hot codes (for inference/acting).
        
        Args:
            z: Continuous latent states [batch_size, z_dim]
            eval_mode: If True, use argmax instead of sampling
            
        Returns:
            code: One-hot codes [batch_size, n_latents * n_classes]
            logp: Log probabilities [batch_size, n_latents]
        """
        batch_size = z.size(0)
        
        # Get logits
        logits = self.encoder(z)
        logits = logits.view(batch_size, self.n_latents, self.n_classes)
        
        # Create categorical distribution
        dist = OneHotCategorical(logits=logits)
        
        if eval_mode:
            # Use argmax for deterministic behavior
            indices = logits.argmax(dim=-1)
            code = F.one_hot(indices, self.n_classes).float()
        else:
            # Sample from distribution
            code = dist.sample()
        
        # Compute log probabilities
        logp = dist.log_prob(code)  # [B, n_latents]
        
        code = code.view(batch_size, -1)  # [B, n_latents * n_classes]
        
        return code, logp
    
    def decode(self, code):
        """
        Decode one-hot codes back to continuous latent states.
        
        Args:
            code: One-hot codes [batch_size, n_latents * n_classes]
            
        Returns:
            z_recon: Reconstructed latent states [batch_size, z_dim]
        """
        return self.decoder(code)
    
    def kl_recon_loss(self, z):
        """
        Compute VAE loss: KL divergence + reconstruction loss.
        
        Args:
            z: Continuous latent states [batch_size, z_dim]
            
        Returns:
            total_loss: Combined KL + reconstruction loss
            recon_loss: Reconstruction loss only (for intrinsic reward)
        """
        batch_size = z.size(0)
        
        # Encode and decode
        code, dist = self.encode(z)
        z_recon = self.decode(code)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(z_recon, z.detach())
        
        # KL divergence with uniform prior
        # For each categorical: KL(q || uniform) = log(n_classes) + E[log q]
        uniform_logprob = -torch.log(torch.tensor(self.n_classes, dtype=torch.float32, device=z.device))
        kl_loss = 0
        for i in range(self.n_latents):
            cat_logits = dist.logits[:, i, :]  # [B, n_classes]
            cat_probs = F.softmax(cat_logits, dim=-1)
            cat_logprobs = F.log_softmax(cat_logits, dim=-1)
            kl_loss += (cat_probs * (cat_logprobs - uniform_logprob)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = kl_loss + self.beta * recon_loss
        
        # Update step counter
        self.steps += 1
        
        return total_loss, recon_loss
    
    def forward(self, z):
        """Forward pass for convenience."""
        code, _ = self.encode(z)
        z_recon = self.decode(code)
        return z_recon, code 