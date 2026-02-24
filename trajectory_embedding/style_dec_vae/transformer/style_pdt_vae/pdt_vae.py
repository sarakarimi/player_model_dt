"""
Style-VAE + Decision Transformer for MiniGrid.

This model learns to:
1. Encode full trajectories (states, actions) into a style latent variable
2. Condition a Decision Transformer on that style to predict actions
"""

import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from envs.three_style_env import MiniGridThreeStyles
from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths
from trajectory_embedding.style_dec_vae.lstm.style_vae import cluster_latents, plot_embeddings
from trajectory_gpt2 import GPT2Model


# =============================================================================
# Dataset
# =============================================================================

class MiniGridDataset(TrajectoryDataset):
    """
    Dataset for style-VAE + Decision Transformer.

    Returns both:
    - Full trajectory: entire episode padded to max_seq_len (for encoder)
    - Context window: random subsequence of length max_len (for decoder)
    """

    def __init__(
        self,
        trajectory_paths,
        max_len=1,
        prob_go_from_end=0,
        pct_traj=1.0,
        rtg_scale=1,
        normalize_state=True,
        preprocess_observations: Callable = None,
        sampling=False,
        index_channel_only=False,
        state_normalization_factor=1,
        action_normalization_factor=1,
        device="cpu",
    ):
        super().__init__(
            trajectory_paths=trajectory_paths,
            max_len=max_len,
            prob_go_from_end=prob_go_from_end,
            pct_traj=pct_traj,
            rtg_scale=rtg_scale,
            normalize_state=normalize_state,
            preprocess_observations=preprocess_observations,
            sampling=sampling,
            index_channel_only=index_channel_only,
            state_normalization_factor=state_normalization_factor,
            action_normalization_factor=action_normalization_factor,
            device=device,
        )

        self.seq_lens = [len(seq) for seq in self.states]
        self.max_seq_len = max(self.seq_lens)

    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):
        """Sample a random context window from the trajectory."""
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # Choose start index
        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None and random.random() < prob_go_from_end:
            si = max(0, traj_rewards.shape[0] - max_len)

        # Slice trajectory
        s = traj_states[si:si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si:si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si:si + max_len].reshape(1, -1, 1)
        d = traj_dones[si:si + max_len].reshape(1, -1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # Pad to max_len
        tlen = s.shape[1]
        padding = max_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)  # -10 for invalid actions
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        # Normalize
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def get_full_traj(self, traj_index):
        """Get the full trajectory padded to max_seq_len."""
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # Full sequence from t=0
        s = traj_states.reshape(1, -1, *self.state_dim)
        a = traj_actions.reshape(1, -1, *self.act_dim)
        r = traj_rewards.reshape(1, -1, 1)
        rtg = traj_rtg.reshape(1, -1, 1)
        d = traj_dones.reshape(1, -1)
        ti = np.arange(0, s.shape[1]).reshape(1, -1)

        # Pad to max_seq_len
        tlen = s.shape[1]
        padding = self.max_seq_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        # Normalize
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        # DT context window
        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end,
        )

        # Full episode for encoder
        full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m = self.get_full_traj(traj_index)

        task_label = self.tasks[traj_index]

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
            full_s = self.preprocess_observations(full_s)

        return (
            s, a, r, d, rtg, ti, m,
            full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m,
            task_label,
        )

    @staticmethod
    def collate_fn(batch):
        """Collate batch into dict format."""
        (
            states, actions, rewards, dones, rtgs, timesteps, masks,
            full_states, full_actions, full_rewards, full_rtgs, full_dones, full_timesteps, full_masks,
            task_labels,
        ) = zip(*batch)

        return {
            # Decoder (context window)
            "states": torch.stack(states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "returns_to_go": torch.stack(rtgs, dim=0),
            "timesteps": torch.stack(timesteps, dim=0),
            "attention_mask": torch.stack(masks, dim=0),
            "dones": torch.stack(dones, dim=0),
            # Encoder (full trajectory)
            "full_states": torch.stack(full_states, dim=0),
            "full_actions": torch.stack(full_actions, dim=0),
            "full_rewards": torch.stack(full_rewards, dim=0),
            "full_returns_to_go": torch.stack(full_rtgs, dim=0),
            "full_timesteps": torch.stack(full_timesteps, dim=0),
            "full_attention_mask": torch.stack(full_masks, dim=0),
            "full_dones": torch.stack(full_dones, dim=0),
            # Labels
            "task_labels": torch.tensor(task_labels, dtype=torch.long),
        }


# =============================================================================
# VAE Utilities
# =============================================================================

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick for VAE."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_diag_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between diagonal Gaussian and standard normal."""
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)


# =============================================================================
# Decision Transformer
# =============================================================================

class DecisionTransformer(nn.Module):
    """
    Decision Transformer that can be conditioned on style tokens.

    Processes sequences as: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
    Optionally prepends style tokens at the beginning for conditioning.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int = None,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        # GPT-2 transformer
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        # Embedding layers
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size)  # Use Embedding for discrete actions
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Prediction heads
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(
        self,
        states: torch.Tensor,           # [B, T, state_dim]
        actions: torch.Tensor,          # [B, T, act_dim] or [B, T, 1]
        returns_to_go: torch.Tensor,    # [B, T, 1]
        timesteps: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor = None,  # [B, T]
        style_tokens: torch.Tensor = None,    # [B, 3, H] optional
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed each modality
        state_embeddings = self.embed_state(states)

        # Embed discrete actions directly
        if actions.ndim == 3:
            actions = actions.squeeze(-1)
        actions = torch.clamp(actions.long(), 0, self.act_dim - 1)  # Handle padding
        action_embeddings = self.embed_action(actions)

        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        returns_embeddings = self.embed_return(returns_to_go)

        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        # Prepend style tokens if provided
        if style_tokens is not None:
            style_tokens_flat = style_tokens.reshape(batch_size, 3, self.hidden_size)
            style_mask = torch.ones(
                (batch_size, 3), dtype=stacked_attention_mask.dtype, device=stacked_attention_mask.device
            )
            stacked_inputs = torch.cat([style_tokens_flat, stacked_inputs], dim=1)
            stacked_attention_mask = torch.cat([style_mask, stacked_attention_mask], dim=1)

        # Run transformer
        x = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )['last_hidden_state']

        # Reshape and extract predictions
        if style_tokens is None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predictions from last seq_length timesteps
        return_preds = self.predict_return(x[:, 2])[:, -seq_length:, :]
        state_preds = self.predict_state(x[:, 2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:, 1])[:, -seq_length:, :]

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        style_tokens: torch.Tensor = None,
    ):
        """Get action prediction for inference."""
        states = states.reshape(1, -1, self.state_dim)
        # For discrete actions, keep shape as [1, T, 1] or [1, T]
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.reshape(1, -1, 1)
        else:
            actions = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # Create attention mask (must be float, not long)
            attention_mask = torch.cat([
                torch.zeros(self.max_length - states.shape[1]),
                torch.ones(states.shape[1])
            ], dim=0).to(dtype=torch.float32, device=states.device).reshape(1, -1)

            # Pad sequences
            pad_len = self.max_length - states.shape[1]
            states = torch.cat([torch.zeros((1, pad_len, self.state_dim), device=states.device), states], dim=1)
            # For discrete actions, match the action tensor shape
            action_pad_shape = (1, pad_len, actions.shape[-1]) if actions.ndim == 3 else (1, pad_len)
            actions = torch.cat([torch.zeros(action_pad_shape, device=actions.device, dtype=actions.dtype), actions], dim=1)
            returns_to_go = torch.cat([torch.zeros((1, pad_len, 1), device=returns_to_go.device), returns_to_go], dim=1)
            timesteps = torch.cat([torch.zeros((1, pad_len), device=timesteps.device), timesteps], dim=1).long()
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, returns_to_go, timesteps,
            attention_mask=attention_mask,
            style_tokens=style_tokens,
        )

        return action_preds[0, -1]


# =============================================================================
# Style VAE + Decision Transformer
# =============================================================================

class StyleVAEPromptDT(nn.Module):
    """
    Style-conditioned Decision Transformer with VAE encoder.

    Architecture:
    1. Encoder: full trajectory (states, actions) -> latent style z
    2. Latent -> 3 style tokens (R, S, A) prepended to DT
    3. Decoder: Decision Transformer conditioned on style tokens
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        latent_dim: int = 32,
        enc_layers: int = 4,
        enc_heads: int = 4,
        enc_ff: int = None,
        enc_dropout: float = 0.1,
        max_length: int = None,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        beta: float = 0.1,
        **dt_kwargs,
    ):
        super().__init__()
        if enc_ff is None:
            enc_ff = 4 * hidden_size

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Decision Transformer (decoder)
        self.dt = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            **dt_kwargs,
        )

        # Trajectory encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=enc_heads,
            dim_feedforward=enc_ff,
            dropout=enc_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.style_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.enc_ln = nn.LayerNorm(hidden_size)

        # VAE layers
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

        # Latent to style tokens
        self.z_to_style_tokens = nn.Sequential(
            nn.Linear(latent_dim, 3 * hidden_size),
            nn.GELU(),
            nn.Linear(3 * hidden_size, 3 * hidden_size),
            nn.LayerNorm(3 * hidden_size),
        )

    def encode_full_trajectory(
        self,
        full_states: torch.Tensor,      # [B, S, state_dim]
        full_actions: torch.Tensor,     # [B, S, act_dim]
        full_timesteps: torch.Tensor,   # [B, S]
        full_attn_mask: torch.Tensor,   # [B, S]
    ):
        """Encode full trajectory into latent style z."""
        batch_size, seq_len, _ = full_states.shape

        # Embed states and actions
        s_emb = self.dt.embed_state(full_states)

        # Embed discrete actions directly
        if full_actions.ndim == 3:
            full_actions = full_actions.squeeze(-1)
        full_actions = torch.clamp(full_actions.long(), 0, self.dt.act_dim - 1)
        a_emb = self.dt.embed_action(full_actions)

        t_emb = self.dt.embed_timestep(full_timesteps)

        # Add time embeddings
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Stack as [s_t, a_t] for each timestep
        tokens = torch.stack((s_emb, a_emb), dim=1)  # [B, 2, S, H]
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_len, self.hidden_size)

        # Token-level attention mask
        token_mask = torch.stack((full_attn_mask, full_attn_mask), dim=1)
        token_mask = token_mask.permute(0, 2, 1).reshape(batch_size, 2 * seq_len)

        # Run encoder
        src_key_padding_mask = (token_mask == 0)  # True = padding
        h = self.style_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        h = self.enc_ln(h)

        # Masked mean pooling
        m = token_mask.unsqueeze(-1).to(h.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (h * m).sum(dim=1) / denom

        # VAE parameters
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        z = reparameterize(mu, logvar)

        return mu, logvar, z

    def latent_to_style_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Convert latent z to style tokens [B, 3, H]."""
        batch_size = z.size(0)
        x = self.z_to_style_tokens(z)
        return x.view(batch_size, 3, self.hidden_size)

    def forward(
        self,
        # Encoder inputs
        full_states: torch.Tensor,
        full_actions: torch.Tensor,
        full_timesteps: torch.Tensor,
        full_attention_mask: torch.Tensor,
        # Decoder inputs
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        # Unused but kept for compatibility
        full_returns_to_go: torch.Tensor = None,
        rewards: torch.Tensor = None,
        prompt=None,
        beta: float = None,
    ):
        if beta is None:
            beta = self.beta

        # Encode trajectory to style
        mu, logvar, z = self.encode_full_trajectory(
            full_states, full_actions, full_timesteps, full_attention_mask
        )
        style_tokens = self.latent_to_style_tokens(z)

        # Run DT conditioned on style
        state_preds, action_preds, return_preds = self.dt(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            style_tokens=style_tokens,
        )

        # Compute KL loss
        kl = kl_diag_gaussian(mu, logvar).mean()
        kl_loss = beta * kl

        return {
            "state_preds": state_preds,
            "action_preds": action_preds,
            "return_preds": return_preds,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "kl_loss": kl_loss,
        }


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_online(
    model: StyleVAEPromptDT,
    dataset: MiniGridDataset,
    num_styles: int = 3,
    num_episodes_per_style: int = 10,
    max_ep_len: int = 100,
    device: str = "cpu",
    initial_rtg: float = 1.0,
    env_kwargs: dict = None,
):
    """
    Evaluate the DT online for each style.

    For each style:
    1. Sample a random trajectory from that style
    2. Encode it to get z
    3. Run DT conditioned on z in the environment
    4. Return mean episode return for each style
    """
    model.eval()

    if env_kwargs is None:
        env_kwargs = {}

    # Map style IDs to style names
    style_names = {0: "bypass", 1: "weapon", 2: "camouflage"}

    # Results: {style_id: [returns]}
    results = {style_id: [] for style_id in range(num_styles)}

    with torch.no_grad():
        for style_id in range(num_styles):
            # Find trajectories of this style
            style_indices = [i for i, label in enumerate(dataset.tasks) if label == style_id]
            if len(style_indices) == 0:
                print(f"Warning: No trajectories found for style {style_id}")
                continue

            # Sample a random trajectory from this style
            traj_idx = random.choice(style_indices)

            # Get full trajectory and encode it
            full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m = dataset.get_full_traj(traj_idx)

            full_s = full_s.unsqueeze(0).to(device)
            full_a = full_a.unsqueeze(0).to(device)
            full_ti = full_ti.unsqueeze(0).to(device)
            full_m = full_m.unsqueeze(0).to(device)

            # Encode to get z
            mu, logvar, z = model.encode_full_trajectory(
                full_states=full_s,
                full_actions=full_a,
                full_timesteps=full_ti,
                full_attn_mask=full_m,
            )

            # Convert z to style tokens
            style_tokens = model.latent_to_style_tokens(z)  # [1, 3, H]

            # Run episodes in the environment
            for ep in range(num_episodes_per_style):
                # Create environment for this style
                env = MiniGridThreeStyles(
                    target_style=style_names[style_id],
                    target_bonus=1.0,
                    non_target_penalty=-1.0,
                    easy_env=False,
                    agent_view_size=3,
                    **env_kwargs
                )

                obs, _ = env.reset(seed=42 + ep)

                # Initialize context - extract only object index channel (channel 0)
                state = torch.from_numpy(obs['image'][:, :, 0].flatten()).float().to(device)
                state_mean = torch.tensor(dataset.state_mean, device=device, dtype=torch.float32)
                state_std = torch.tensor(dataset.state_std, device=device, dtype=torch.float32)
                state = (state - state_mean) / state_std

                states = state.reshape(1, 1, -1)  # [1, 1, state_dim]
                # Use action 0 as padding token, matching the sequence length of states
                actions = torch.zeros((1, 1, 1), dtype=torch.long, device=device)  # [1, 1, 1]
                rtgs = torch.tensor([[[initial_rtg]]], dtype=torch.float32, device=device)  # [1, 1, 1]
                timesteps = torch.tensor([[0]], dtype=torch.long, device=device)  # [1, 1]

                episode_return = 0
                done = False
                t = 0

                while not done and t < max_ep_len:
                    # Get action from DT (use forward directly without padding logic)
                    with torch.no_grad():
                        # Create attention mask - all ones (attend to all positions)
                        attn_mask = torch.ones((1, states.shape[1]), dtype=torch.float32, device=device)

                        _, action_preds, _ = model.dt.forward(
                            states=states,
                            actions=actions,
                            returns_to_go=rtgs,
                            timesteps=timesteps,
                            attention_mask=attn_mask,
                            style_tokens=style_tokens,
                        )

                    action = torch.argmax(action_preds[:, -1], dim=-1).item()

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    episode_return += reward
                    t += 1

                    if not done:
                        # Update context - extract only object index channel (channel 0)
                        next_state = torch.from_numpy(next_obs['image'][:, :, 0].flatten()).float().to(device)
                        next_state = (next_state - state_mean) / state_std

                        states = torch.cat([states, next_state.reshape(1, 1, -1)], dim=1)
                        actions = torch.cat([actions, torch.tensor([[[action]]], dtype=torch.long, device=device)], dim=1)

                        # Update RTG
                        next_rtg = rtgs[:, -1:, :] - reward
                        rtgs = torch.cat([rtgs, next_rtg], dim=1)

                        timesteps = torch.cat([timesteps, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)

                        # Truncate to max context window to avoid memory issues
                        max_context = 20  # Keep last 20 timesteps
                        if states.shape[1] > max_context:
                            states = states[:, -max_context:]
                            actions = actions[:, -max_context:]
                            rtgs = rtgs[:, -max_context:]
                            timesteps = timesteps[:, -max_context:]

                results[style_id].append(episode_return)
                env.close()

            print(f"Style {style_id}: mean return = {np.mean(results[style_id]):.3f} ± {np.std(results[style_id]):.3f}")

    model.train()
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_eval_results(eval_history: dict, save_path: str = "eval_results.png"):
    """Plot the online evaluation results for each style over training."""
    epochs = eval_history["epochs"]
    style_names = {0: "Bypass", 1: "Weapon", 2: "Camouflage"}

    # Combined plot with all styles
    plt.figure(figsize=(10, 6))
    for style_id in range(3):
        returns = eval_history[f"style_{style_id}"]
        plt.plot(epochs, returns, marker='o', label=f"{style_names[style_id]} (Style {style_id})", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Episode Return", fontsize=12)
    plt.title("Online Evaluation: Episode Returns by Style", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    print(f"Saved combined evaluation plot to {save_path}")
    plt.close()

    # Individual plots for each style
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for style_id in range(3):
        ax = axes[style_id]
        returns = eval_history[f"style_{style_id}"]

        ax.plot(epochs, returns, marker='o', color=f'C{style_id}', linewidth=2, markersize=6)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Mean Episode Return", fontsize=11)
        ax.set_title(f"{style_names[style_id]} (Style {style_id})", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add min/max/final annotations
        if returns:
            final_return = returns[-1]
            max_return = max(returns)
            ax.axhline(y=max_return, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.02, 0.98, f"Final: {final_return:.2f}\nMax: {max_return:.2f}",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    individual_path = save_path.replace('.png', '_individual.png')
    plt.savefig(individual_path, dpi=150)
    print(f"Saved individual style plots to {individual_path}")
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train_style_prompt_dt(
    model: StyleVAEPromptDT,
    dataloader: DataLoader,
    num_epochs: int,
    device: str = "cpu",
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    action_loss_weight: float = 1.0,
    log_every: int = 10,
    save_path: str = None,
    eval_every: int = 10,
    eval_episodes_per_style: int = 10,
    max_ep_len: int = 100,
    initial_rtg: float = 1.0,
):
    """Train the Style-VAE + Decision Transformer model."""
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    # Track evaluation results
    eval_history = {
        "epochs": [],
        "style_0": [],
        "style_1": [],
        "style_2": [],
    }

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_bc = 0.0
        running_kl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtgs = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            full_states = batch["full_states"].to(device)
            full_actions = batch["full_actions"].to(device)
            full_timesteps = batch["full_timesteps"].to(device)
            full_attn_mask = batch["full_attention_mask"].to(device)

            # Forward pass
            out = model(
                full_states=full_states,
                full_actions=full_actions,
                full_timesteps=full_timesteps,
                full_attention_mask=full_attn_mask,
                states=states,
                actions=actions,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
            )

            action_preds = out["action_preds"]  # [B, K, num_classes]
            kl_loss = out["kl_loss"]

            # Action BC loss (cross-entropy)
            if action_loss_weight > 0:
                batch_size, seq_len, num_classes = action_preds.shape

                # Prepare actions for cross-entropy
                if actions.ndim == 3:
                    actions = actions.squeeze(-1)
                actions = torch.clamp(actions.long(), 0, num_classes - 1)

                # Compute masked cross-entropy loss
                action_preds_flat = action_preds.reshape(batch_size * seq_len, num_classes)
                actions_flat = actions.reshape(batch_size * seq_len)

                ce_loss = torch.nn.functional.cross_entropy(
                    action_preds_flat, actions_flat, reduction='none'
                ).reshape(batch_size, seq_len)

                # Apply mask
                valid_mask = attn_mask.to(ce_loss.dtype)
                action_bc = (ce_loss * valid_mask).sum()
            else:
                action_bc = torch.zeros((), device=device)

            # Total loss
            loss = action_loss_weight * action_bc + kl_loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Logging
            running_loss += loss.item()
            running_bc += action_bc.item()
            running_kl += kl_loss.item()
            n_batches += 1

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(dataloader)} "
                    f"| loss={loss.item():.6f} bc={action_bc.item():.6f} kl={kl_loss.item():.6f}"
                )

        # End of epoch
        print(
            f"===> Epoch {epoch+1}/{num_epochs} "
            f"| avg_loss={running_loss/n_batches:.6f} "
            f"| avg_bc={running_bc/n_batches:.6f} "
            f"| avg_kl={running_kl/n_batches:.6f}"
        )

        # Online evaluation every eval_every epochs
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            print(f"\n=== Online Evaluation at Epoch {epoch+1} ===")
            eval_results = evaluate_online(
                model=model,
                dataset=dataloader.dataset,
                num_styles=3,
                num_episodes_per_style=eval_episodes_per_style,
                max_ep_len=max_ep_len,
                device=device,
                initial_rtg=initial_rtg,
            )

            # Record results
            eval_history["epochs"].append(epoch + 1)
            for style_id in range(3):
                mean_return = np.mean(eval_results[style_id]) if eval_results[style_id] else 0.0
                eval_history[f"style_{style_id}"].append(mean_return)

            print()

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    # Plot evaluation results
    if eval_history["epochs"]:
        plot_eval_results(eval_history, save_path="plots/eval_results.png")

    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    dataset_params = {
        "sampling": False,
        "index_channel_only": True,
        "state_normalization_factor": 1, #9,
        "action_normalization_factor": 1, #6,
    }
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    # Create model
    model = StyleVAEPromptDT(
        state_dim=9,
        act_dim=7,
        hidden_size=128,
        latent_dim=16,
        max_length=20,
        max_ep_len=100,
        action_tanh=False,
        beta=0.0085,
        n_layer=4,
        n_head=8,
    )

    # Train
    train_style_prompt_dt(
        model=model,
        dataloader=loader,
        num_epochs=100,
        device=device,
        lr=1e-3,
        grad_clip=1.0,
        action_loss_weight=1.0,
        log_every=10,
        save_path="style_prompt_dt_minigrid.pth",
        eval_every=10,
        eval_episodes_per_style=10,
        max_ep_len=100,
        initial_rtg=1.0,
    )

    # Evaluate: extract latents and visualize
    model.eval()
    Z, y_true = [], []

    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtgs = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            full_states = batch["full_states"].to(device)
            full_actions = batch["full_actions"].to(device)
            full_timesteps = batch["full_timesteps"].to(device)
            full_attn_mask = batch["full_attention_mask"].to(device)

            out = model(
                full_states=full_states,
                full_actions=full_actions,
                full_timesteps=full_timesteps,
                full_attention_mask=full_attn_mask,
                states=states,
                actions=actions,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
            )

            Z.append(out["z"].cpu())
            y_true.extend(batch["task_labels"].cpu().numpy().tolist())

    # Cluster and plot
    Z = torch.cat(Z, 0).cpu().numpy()
    predicted_labels, _ = cluster_latents(Z, 3)

    plot_embeddings(gtruth=predicted_labels, Z=Z, label_name="task_predicted")
    plot_embeddings(gtruth=y_true, Z=Z, label_name="task_ground_truth")