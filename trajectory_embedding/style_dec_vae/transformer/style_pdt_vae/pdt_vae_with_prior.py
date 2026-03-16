
import random
from typing import Callable, Tuple, Dict, Optional

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
    Returns:
    - DT context window (states/actions/rtg/timesteps/mask)
    - Full episode for encoder
    - controls: per-trajectory designer controls (float vector)
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
        control_dim: int = 5,
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
        self.control_dim = control_dim


    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None and random.random() < prob_go_from_end:
            si = max(0, traj_rewards.shape[0] - max_len)

        s = traj_states[si:si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si:si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si:si + max_len].reshape(1, -1, 1)
        d = traj_dones[si:si + max_len].reshape(1, -1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        tlen = s.shape[1]
        padding = max_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def get_full_traj(self, traj_index):
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        s = traj_states.reshape(1, -1, *self.state_dim)
        a = traj_actions.reshape(1, -1, *self.act_dim)
        r = traj_rewards.reshape(1, -1, 1)
        rtg = traj_rtg.reshape(1, -1, 1)
        d = traj_dones.reshape(1, -1)
        ti = np.arange(0, s.shape[1]).reshape(1, -1)

        tlen = s.shape[1]
        padding = self.max_seq_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end,
        )

        full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m = self.get_full_traj(traj_index)

        task_label = self.tasks[traj_index]

        controls = torch.tensor(self.controls[traj_index], dtype=torch.float32)  # [control_dim]

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
            full_s = self.preprocess_observations(full_s)

        return (
            s, a, r, d, rtg, ti, m,
            full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m,
            controls,
            task_label,
        )

    @staticmethod
    def collate_fn(batch):
        (
            states, actions, rewards, dones, rtgs, timesteps, masks,
            full_states, full_actions, full_rewards, full_rtgs, full_dones, full_timesteps, full_masks,
            controls,
            task_labels,
        ) = zip(*batch)

        return {
            "states": torch.stack(states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "returns_to_go": torch.stack(rtgs, dim=0),
            "timesteps": torch.stack(timesteps, dim=0),
            "attention_mask": torch.stack(masks, dim=0),
            "dones": torch.stack(dones, dim=0),

            "full_states": torch.stack(full_states, dim=0),
            "full_actions": torch.stack(full_actions, dim=0),
            "full_rewards": torch.stack(full_rewards, dim=0),
            "full_returns_to_go": torch.stack(full_rtgs, dim=0),
            "full_timesteps": torch.stack(full_timesteps, dim=0),
            "full_attention_mask": torch.stack(full_masks, dim=0),
            "full_dones": torch.stack(full_dones, dim=0),

            "controls": torch.stack(controls, dim=0),  # [B, control_dim] float32
            "task_labels": torch.tensor(task_labels, dtype=torch.long),
        }


# =============================================================================
# VAE Utilities
# =============================================================================

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_q_p_diag(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term = (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8)
    kl = 0.5 * torch.sum(logvar_p - logvar_q + term - 1.0, dim=-1)
    return kl  # [B]


# =============================================================================
# Conditional Prior on Controls
# =============================================================================

class ConditionalPrior(nn.Module):
    def __init__(self, control_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(control_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        nn.init.constant_(self.to_logvar.bias, -3.0)

    def forward(self, controls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(controls)
        return self.to_mu(h), self.to_logvar(h)


# =============================================================================
# Decision Transformer
# Code taken from the Prompt-dt repository
# =============================================================================

class DecisionTransformer(nn.Module):
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

        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor = None,
        style_tokens: torch.Tensor = None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        state_embeddings = self.embed_state(states)

        if actions.ndim == 3:
            actions = actions.squeeze(-1)
        actions = torch.clamp(actions.long(), 0, self.act_dim - 1)
        action_embeddings = self.embed_action(actions)

        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        returns_embeddings = self.embed_return(returns_to_go)

        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        if style_tokens is not None:
            style_tokens_flat = style_tokens.reshape(batch_size, 3, self.hidden_size)
            style_mask = torch.ones((batch_size, 3), dtype=stacked_attention_mask.dtype, device=stacked_attention_mask.device)
            stacked_inputs = torch.cat([style_tokens_flat, stacked_inputs], dim=1)
            stacked_attention_mask = torch.cat([style_mask, stacked_attention_mask], dim=1)

        x = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask)['last_hidden_state']

        if style_tokens is None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])[:, -seq_length:, :]
        state_preds = self.predict_state(x[:, 2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:, 1])[:, -seq_length:, :]

        return state_preds, action_preds, return_preds


# =============================================================================
# Style VAE + DT (Conditional Prior on Controls)
# =============================================================================

class StyleVAEPromptDT(nn.Module):
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
        control_dim: int = 5,
        prior_hidden: int = 128,
        free_bits: float = 0.0,
        **dt_kwargs,
    ):
        super().__init__()
        if enc_ff is None:
            enc_ff = 4 * hidden_size

        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.free_bits = free_bits

        self.dt = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            **dt_kwargs,
        )

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

        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

        self.z_to_style_tokens = nn.Sequential(
            nn.Linear(latent_dim, 3 * hidden_size),
            # nn.GELU(),
            # nn.Linear(3 * hidden_size, 3 * hidden_size),
            # nn.LayerNorm(3 * hidden_size),
        )

        self.prior = ConditionalPrior(control_dim=control_dim, latent_dim=latent_dim, hidden=prior_hidden)

    def encode_full_trajectory(
        self,
        full_states: torch.Tensor,
        full_actions: torch.Tensor,
        full_timesteps: torch.Tensor,
        full_attn_mask: torch.Tensor,
    ):
        batch_size, seq_len, _ = full_states.shape

        s_emb = self.dt.embed_state(full_states)

        if full_actions.ndim == 3:
            full_actions = full_actions.squeeze(-1)
        full_actions = torch.clamp(full_actions.long(), 0, self.dt.act_dim - 1)
        a_emb = self.dt.embed_action(full_actions)

        t_emb = self.dt.embed_timestep(full_timesteps)
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        tokens = torch.stack((s_emb, a_emb), dim=1)  # [B,2,S,H]
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_len, self.hidden_size)

        token_mask = torch.stack((full_attn_mask, full_attn_mask), dim=1)
        token_mask = token_mask.permute(0, 2, 1).reshape(batch_size, 2 * seq_len)

        src_key_padding_mask = (token_mask == 0)
        h = self.style_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        h = self.enc_ln(h)

        m = token_mask.unsqueeze(-1).to(h.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (h * m).sum(dim=1) / denom

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        z = reparameterize(mu, logvar)
        return mu, logvar, z

    def latent_to_style_tokens(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        x = self.z_to_style_tokens(z)
        return x.view(batch_size, 3, self.hidden_size)

    def sample_z_from_prior(self, controls: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu_p, logvar_p = self.prior(controls)
        if deterministic:
            return mu_p
        return reparameterize(mu_p, logvar_p)

    def forward(
        self,
        full_states: torch.Tensor,
        full_actions: torch.Tensor,
        full_timesteps: torch.Tensor,
        full_attention_mask: torch.Tensor,
        controls: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        full_returns_to_go: torch.Tensor = None,
        rewards: torch.Tensor = None,
        prompt=None,
        beta: float = None,
    ):
        if beta is None:
            beta = self.beta

        mu_q, logvar_q, z = self.encode_full_trajectory(full_states, full_actions, full_timesteps, full_attention_mask)
        mu_p, logvar_p = self.prior(controls)

        style_tokens = self.latent_to_style_tokens(z)

        state_preds, action_preds, return_preds = self.dt(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            style_tokens=style_tokens,
        )

        kl_per = kl_q_p_diag(mu_q, logvar_q, mu_p, logvar_p)
        if self.free_bits > 0:
            kl_per = torch.clamp(kl_per, min=self.free_bits)
        kl_loss = beta * kl_per.mean()

        return {
            "state_preds": state_preds,
            "action_preds": action_preds,
            "return_preds": return_preds,
            "mu": mu_q,
            "logvar": logvar_q,
            "z": z,
            "mu_prior": mu_p,
            "logvar_prior": logvar_p,
            "kl_loss": kl_loss,
        }

# =============================================================================
# Training (minimal changes: pass controls)
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
    beta_warmup_epochs: int = 0,
):
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
        if beta_warmup_epochs > 0:
            beta = model.beta * min(1.0, (epoch + 1) / beta_warmup_epochs)
        else:
            beta = model.beta

        running_loss = 0.0
        running_bc = 0.0
        running_kl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtgs = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            full_states = batch["full_states"].to(device)
            full_actions = batch["full_actions"].to(device)
            full_timesteps = batch["full_timesteps"].to(device)
            full_attn_mask = batch["full_attention_mask"].to(device)

            controls = batch["controls"].to(device)  # [B, control_dim]

            out = model(
                full_states=full_states,
                full_actions=full_actions,
                full_timesteps=full_timesteps,
                full_attention_mask=full_attn_mask,
                controls=controls,
                states=states,
                actions=actions,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
                beta=beta,
            )

            action_preds = out["action_preds"]
            kl_loss = out["kl_loss"]

            if action_loss_weight > 0:
                B, T, C = action_preds.shape
                if actions.ndim == 3:
                    actions_ce = actions.squeeze(-1)
                else:
                    actions_ce = actions
                actions_ce = torch.clamp(actions_ce.long(), 0, C - 1)

                logits = action_preds.reshape(B * T, C)
                targets = actions_ce.reshape(B * T)

                ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none").reshape(B, T)
                valid = attn_mask.to(ce.dtype)
                action_bc = (ce * valid).sum()
            else:
                action_bc = torch.zeros((), device=device)

            loss = action_loss_weight * action_bc + kl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            running_bc += float(action_bc.item())
            running_kl += float(kl_loss.item())
            n_batches += 1

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(dataloader)} "
                    f"| loss={loss.item():.6f} bc={action_bc.item():.6f} kl={kl_loss.item():.6f}"
                )

        print(
            f"===> Epoch {epoch+1}/{num_epochs} "
            f"| avg_loss={running_loss/n_batches:.6f} "
            f"| avg_bc={running_bc/n_batches:.6f} "
            f"| avg_kl={running_kl/n_batches:.6f} "
            f"| beta={beta:.6f}"
        )

        # Online evaluation every eval_every epochs
        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            print(f"\n=== Online Evaluation at Epoch {epoch + 1} ===")
            eval_results = evaluate_online_controls(
                model=model,
                dataset=dataloader.dataset,
                num_styles=3,
                num_episodes_per_style=eval_episodes_per_style,
                max_ep_len=max_ep_len,
                device=device,
                initial_rtg=initial_rtg,
                env_kwargs=None,
                deterministic_prior=False,
                max_context=20,
            )


            # Record results
            eval_history["epochs"].append(epoch + 1)
            for style_id in range(3):
                mean_return = np.mean(eval_results[style_id]) if eval_results[style_id] else 0.0
                eval_history[f"style_{style_id}"].append(mean_return)

            print()

        if save_path is not None and (epoch + 1) % eval_every == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    # Plot evaluation results
    if eval_history["epochs"]:
        plot_eval_results(eval_history, save_path="plots/eval_results_style_dt.png")

    return model


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_online_controls(
    model: StyleVAEPromptDT,
    dataset: MiniGridDataset,
    num_styles: int = 3,
    num_episodes_per_style: int = 10,
    max_ep_len: int = 100,
    device: str = "cpu",
    initial_rtg: float = 1.0,
    env_kwargs: dict = None,
    deterministic_prior: bool = True,
    max_context: int = 20,
):
    """
    Online evaluation for the *controls-conditioned prior* model.

    For each style (only for evaluation bookkeeping):
      1) Choose a control vector c (designer controls)
      2) Sample z ~ p(z|c) using the learned prior (NO reference trajectory needed)
      3) Convert z -> style_tokens
      4) Rollout DT conditioned on style_tokens

    Returns:
      results: dict {style_id: [episode_returns]}
    """
    model.eval()
    if env_kwargs is None:
        env_kwargs = {}

    # style id -> env target style name (only for spawning env variants)
    style_names = {0: "bypass", 1: "weapon", 2: "camouflage"}

    # [risk_tolerance, resource_pref, stealth_pref, safety_pref, commitment]
    # Fallback when dataset has no controls stored (should not happen with new datasets).
    fallback_style_to_controls = {
        0: np.array([0.67, 0.01, 0.53, 0.53, 0.82], dtype=np.float32),  # bypass
        1: np.array([0.92, 0.51, 0.00, 0.00, 0.59], dtype=np.float32),  # weapon
        2: np.array([0.92, 0.53, 1.00, 0.63, 0.74], dtype=np.float32),  # camouflage
    }

    results = {style_id: [] for style_id in range(num_styles)}

    # state normalization tensors
    state_mean = torch.tensor(dataset.state_mean, device=device, dtype=torch.float32)
    state_std = torch.tensor(dataset.state_std, device=device, dtype=torch.float32)

    with torch.no_grad():
        for style_id in range(num_styles):
            c = None

            # if hasattr(dataset, "controls") and dataset.controls is not None:
            # sample one trajectory index of this style and take its controls
            style_indices = [i for i, label in enumerate(dataset.tasks) if label == style_id]
            if len(style_indices) > 0:
                traj_idx = random.choice(style_indices)
                c = dataset.controls[traj_idx]  # numpy array [control_dim]
                c = np.asarray(c, dtype=np.float32)

            # if c is None:
            #     c = fallback_style_to_controls[style_id]

            controls = torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)

            # sample z from prior given controls
            z = model.sample_z_from_prior(controls, deterministic=deterministic_prior)
            style_tokens = model.latent_to_style_tokens(z)

            # rollout episodes
            for ep in range(num_episodes_per_style):
                env = MiniGridThreeStyles(
                    target_style=style_names[style_id],
                    target_bonus=1.0,
                    non_target_penalty=-1.0,
                    easy_env=False,
                    agent_view_size=3,
                    randomize_layout=True,
                    **env_kwargs
                )

                obs, _ = env.reset(seed=42 + ep)
                # env.render()

                # state: object index channel 0, flattened, normalized like training
                state = torch.from_numpy(obs["image"][:, :, 0].flatten()).float().to(device)
                state = (state - state_mean) / state_std

                states = state.reshape(1, 1, -1)
                actions = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
                rtgs = torch.tensor([[[initial_rtg]]], dtype=torch.float32, device=device)
                timesteps = torch.tensor([[0]], dtype=torch.long, device=device)

                episode_return = 0.0
                done = False
                t = 0

                while not done and t < max_ep_len:
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

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    episode_return += float(reward)
                    t += 1

                    if not done:
                        next_state = torch.from_numpy(next_obs["image"][:, :, 0].flatten()).float().to(device)
                        next_state = (next_state - state_mean) / state_std

                        states = torch.cat([states, next_state.reshape(1, 1, -1)], dim=1)
                        actions = torch.cat(
                            [actions, torch.tensor([[[action]]], dtype=torch.long, device=device)],
                            dim=1,
                        )

                        next_rtg = rtgs[:, -1:, :] - reward
                        rtgs = torch.cat([rtgs, next_rtg], dim=1)

                        timesteps = torch.cat(
                            [timesteps, torch.tensor([[t]], dtype=torch.long, device=device)],
                            dim=1,
                        )

                        # keep only last max_context steps
                        if max_context is not None and states.shape[1] > max_context:
                            states = states[:, -max_context:]
                            actions = actions[:, -max_context:]
                            rtgs = rtgs[:, -max_context:]
                            timesteps = timesteps[:, -max_context:]

                results[style_id].append(episode_return)
                env.close()

            print(
                f"[controls->prior] Style {style_id} ({style_names[style_id]}): "
                f"mean return = {np.mean(results[style_id]):.3f} ± {np.std(results[style_id]):.3f} "
                f"| deterministic_prior={deterministic_prior}"
            )

    model.train()
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_eval_results(eval_history: dict, save_path: str = "style_dt_eval_results.png"):
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

    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    max_len = 8
    control_dim = 3
    dataset_params = {
        "sampling": True,
        "index_channel_only": True,
        "state_normalization_factor": 1,
        "action_normalization_factor": 1,
        "max_len": max_len,
        "control_dim": control_dim, #5,
    }
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    model = StyleVAEPromptDT(
        state_dim=9,
        act_dim=7,
        hidden_size=128,
        latent_dim=16,
        max_length=max_len,
        max_ep_len=100,
        action_tanh=False,
        beta=0.0085,
        control_dim=control_dim,
        prior_hidden=128,
        free_bits=0.0,
        n_layer=4,
        n_head=8,
    )

    train_style_prompt_dt(
        model=model,
        dataloader=loader,
        num_epochs=100,
        device=device,
        lr=1e-3,
        grad_clip=1.0,
        action_loss_weight=1.0,
        log_every=10,
        save_path="trained_models/style_prompt_dt_minigrid_controls_condprior.pth",
        beta_warmup_epochs=0
    )

    # Latent visualization (still uses encoder z)
    model.eval()
    Z, y_true = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                full_states=batch["full_states"].to(device),
                full_actions=batch["full_actions"].to(device),
                full_timesteps=batch["full_timesteps"].to(device),
                full_attention_mask=batch["full_attention_mask"].to(device),
                controls=batch["controls"].to(device),
                states=batch["states"].to(device),
                actions=batch["actions"].to(device),
                returns_to_go=batch["returns_to_go"].to(device),
                timesteps=batch["timesteps"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            Z.append(out["z"].cpu())
            y_true.extend(batch["task_labels"].cpu().numpy().tolist())

    Z = torch.cat(Z, 0).cpu().numpy()
    predicted_labels, _ = cluster_latents(Z, 3)
    plot_embeddings(gtruth=predicted_labels, Z=Z, label_name="task_predicted")
    plot_embeddings(gtruth=y_true, Z=Z, label_name="task_ground_truth")
