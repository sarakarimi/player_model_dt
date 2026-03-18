"""
Control-Conditioned Decision Transformer (ControlDT) for the MiniGrid
three-style env.

The model is conditioned on a 5-dim designer control vector that is embedded
as style tokens prepended to the (rtg, state, action) sequence.  This is the
"oracle control" baseline: it receives the intended behavioral parameters
directly, without a reference trajectory or a VAE encoder.

Architecture vs. pdt_vae_with_prior.py:
  - Same GPT2 backbone and DT decoder
  - Same 3-token style-token prepending scheme
  - Control vector → 3 style tokens via a 2-layer MLP (mirrors z_to_style_tokens)
  - No encoder, no VAE, no KL loss — BC loss only

Mirrors prompt_dt.py in dataset, training loop, and evaluation structure for
direct comparison.

Control vector dims (same as pdt_vae_with_prior.py):
    [risk_tolerance, resource_pref, stealth_pref, safety_pref, commitment]
"""

import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from envs.three_style_env import MiniGridThreeStyles
from style_decision_transformer import paths
from trajectory_gpt2 import GPT2Model


# ---------------------------------------------------------------------------
# Control vector metadata
# ---------------------------------------------------------------------------

CONTROL_NAMES = ["risk_tolerance", "resource_pref", "commitment"] #, "stealth_pref", "safety_pref", "commitment"]
CONTROL_DIM   = len(CONTROL_NAMES)

STYLE_NAMES = {0: "bypass", 1: "weapon", 2: "camouflage"}


# =============================================================================
# Dataset
# =============================================================================

class ControlDataset(TrajectoryDataset):
    """
    TrajectoryDataset augmented with per-trajectory control vectors.

    Controls are derived from episode_summary (same logic as MiniGridDataset in
    pdt_vae_with_prior.py), with a fallback to canonical style-level vectors
    when episode_summary is unavailable.
    """

    def __init__(
        self,
        trajectory_paths,
        max_len: int = 20,
        prob_go_from_end: float = 0,
        pct_traj: float = 1.0,
        rtg_scale: float = 1,
        normalize_state: bool = True,
        preprocess_observations: Callable = None,
        sampling: bool = False,
        index_channel_only: bool = False,
        state_normalization_factor: float = 1,
        action_normalization_factor: float = 1,
        device: str = "cpu",
        control_dim: int = CONTROL_DIM,
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
        self.control_dim = control_dim

    # ------------------------------------------------------------------
    def get_traj(self, traj_index: int, max_len: int = 100, prob_go_from_end=None):
        traj_rewards = self.rewards[traj_index]
        traj_states  = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones   = self.dones[traj_index]
        traj_rtg     = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None and random.random() < prob_go_from_end:
            si = max(0, traj_rewards.shape[0] - max_len)

        s   = traj_states[si:si + max_len].reshape(1, -1, *self.state_dim)
        a   = traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim)
        r   = traj_rewards[si:si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si:si + max_len].reshape(1, -1, 1)
        d   = traj_dones[si:si + max_len].reshape(1, -1)
        ti  = np.arange(si, si + s.shape[1]).reshape(1, -1)

        tlen    = s.shape[1]
        padding = max_len - tlen
        s   = self.add_padding(s,   0,         padding)
        a   = self.add_padding(a,   -10,        padding)
        r   = self.add_padding(r,   0,          padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d   = self.add_padding(d,   2,          padding)
        ti  = self.add_padding(ti,  0,          padding)
        m   = self.add_padding(np.ones((1, tlen)), 0, padding)

        s   = (s   - self.state_mean) / self.state_std
        rtg = rtg  / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        traj_index = self.indices[idx]
        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end,
        )
        control = torch.tensor(self.controls[traj_index], dtype=torch.float32)
        task_label = int(self.tasks[traj_index])

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)

        return s, a, r, d, rtg, ti, m, control, task_label

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        (
            states, actions, rewards, dones, rtgs, timesteps, masks,
            controls, task_labels,
        ) = zip(*batch)

        return {
            "states":         torch.stack(states,    dim=0),
            "actions":        torch.stack(actions,   dim=0),
            "rewards":        torch.stack(rewards,   dim=0),
            "returns_to_go":  torch.stack(rtgs,      dim=0),
            "timesteps":      torch.stack(timesteps, dim=0),
            "attention_mask": torch.stack(masks,     dim=0),
            "dones":          torch.stack(dones,     dim=0),
            "controls":       torch.stack(controls,  dim=0),
            "task_labels":    torch.tensor(task_labels, dtype=torch.long),
        }


# =============================================================================
# Control-Conditioned Decision Transformer
# =============================================================================

class ControlConditionedDT(nn.Module):
    """
    Decision Transformer conditioned on a designer control vector.

    The control vector (control_dim floats in [0,1]) is projected to 3 style
    tokens via a 2-layer MLP and prepended before the (rtg, state, action)
    sequence — exactly matching the style-token scheme in pdt_vae_with_prior.py:

        [ctrl_tok_0, ctrl_tok_1, ctrl_tok_2 | rtg_1, s_1, a_1, rtg_2, ...]

    Using 3 tokens (same as the latent_dim of the VAE) keeps the architecture
    identical to pdt_vae_with_prior.py while bypassing the encoder entirely.

    Note: actions are discrete integers → nn.Embedding (MiniGrid, 7 actions).
    """

    NUM_STYLE_TOKENS = 3  # matches latent_dim in pdt_vae_with_prior.py

    def __init__(
        self,
        state_dim:   int,
        act_dim:     int,
        hidden_size: int,
        control_dim: int = CONTROL_DIM,
        max_length:  Optional[int] = None,
        max_ep_len:  int = 4096,
        action_tanh: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size
        self.control_dim = control_dim
        self.max_length  = max_length

        config = transformers.GPT2Config(
            vocab_size=1, n_embd=hidden_size, **kwargs
        )
        self.transformer = GPT2Model(config)

        # Current-context embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return   = nn.Linear(1, hidden_size)
        self.embed_state    = nn.Linear(state_dim, hidden_size)
        self.embed_action   = nn.Embedding(act_dim, hidden_size)
        self.embed_ln       = nn.LayerNorm(hidden_size)

        # Control vector → style tokens (mirrors z_to_style_tokens in VAE model)
        self.control_to_style_tokens = nn.Sequential(
            nn.Linear(control_dim, self.NUM_STYLE_TOKENS * hidden_size),
            # nn.GELU(),
            # nn.Linear(self.NUM_STYLE_TOKENS * hidden_size, self.NUM_STYLE_TOKENS * hidden_size),
        )

        self.predict_state  = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def controls_to_style_tokens(self, controls: torch.Tensor) -> torch.Tensor:
        """controls: [B, control_dim] → style_tokens: [B, NUM_STYLE_TOKENS, H]"""
        return self.control_to_style_tokens(controls).view(
            controls.size(0), self.NUM_STYLE_TOKENS, self.hidden_size
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        controls,
        attention_mask=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        # ------ embed current context (rtg, state, action) -------------------
        acts = torch.clamp(actions.long().squeeze(-1), 0, self.act_dim - 1)
        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        time_emb   = self.embed_timestep(timesteps)
        state_emb  = self.embed_state(states)          + time_emb
        action_emb = self.embed_action(acts)           + time_emb
        return_emb = self.embed_return(returns_to_go)  + time_emb

        stacked_inputs = (
            torch.stack((return_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # ------ embed control → style tokens, prepend ----------------------
        style_tokens = self.controls_to_style_tokens(controls) # [B, 3, H]
        style_mask   = torch.ones(
            (batch_size, self.NUM_STYLE_TOKENS),
            dtype=stacked_mask.dtype, device=states.device
        )

        # [B, 3 + 3T, H]
        all_tokens = torch.cat([style_tokens, stacked_inputs], dim=1)
        all_mask   = torch.cat([style_mask,   stacked_mask],   dim=1)

        x = self.transformer(
            inputs_embeds=all_tokens, attention_mask=all_mask
        )["last_hidden_state"]    # [B, 3+3T, H]

        # Reshape treating every 3 consecutive tokens as one timestep.
        # The 3 style tokens occupy "timestep 0"; actual timesteps follow.
        # [:, -seq_length:, :] extracts only the current-context positions.
        x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])[:, -seq_length:, :]
        state_preds  = self.predict_state(x[:, 2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:, 1])[:, -seq_length:, :]

        return state_preds, action_preds, return_preds

    # ------------------------------------------------------------------
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, controls, **kwargs):
        """
        Inference helper: pads/trims context to max_length, returns action
        logits for the last timestep.
        """
        states        = states.reshape(1, -1, self.state_dim)
        actions       = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps     = timesteps.reshape(1, -1)
        controls      = controls.reshape(1, self.control_dim)

        if self.max_length is not None:
            states        = states[:, -self.max_length:]
            actions       = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps     = timesteps[:, -self.max_length:]

            pad = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(pad), torch.ones(states.shape[1])]
            ).to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((1, pad, self.state_dim), device=states.device), states], dim=1
            ).float()
            actions = torch.cat(
                [torch.zeros((1, pad), dtype=torch.long, device=actions.device), actions], dim=1
            )
            returns_to_go = torch.cat(
                [torch.zeros((1, pad, 1), device=returns_to_go.device), returns_to_go], dim=1
            ).float()
            timesteps = torch.cat(
                [torch.zeros((1, pad), dtype=torch.long, device=timesteps.device), timesteps], dim=1
            )
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, controls,
            attention_mask=attention_mask, **kwargs
        )
        return action_preds[0, -1]


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_online_control(
    eval_model:             ControlConditionedDT,
    eval_dataset:           ControlDataset,
    num_styles:             int   = 3,
    num_episodes_per_style: int   = 10,
    max_ep_len:             int   = 100,
    eval_device:            str   = "cpu",
    initial_rtg:            float = 1.0,
    env_kwargs:             dict  = None,
):
    """
    For each style, roll out the model conditioned on that style's canonical
    control vector (STYLE_CONTROLS).

    Returns:
        results: {style_id: [episode_returns]}
    """
    eval_model.eval()
    if env_kwargs is None:
        env_kwargs = {}

    # [risk_tolerance, resource_pref, stealth_pref, safety_pref, commitment]
    # Fallback when dataset has no controls stored (should not happen with new datasets).
    fallback_style_to_controls = {
        0: np.array([0.67, 0.01, 0.53, 0.53, 0.82], dtype=np.float32),  # bypass
        1: np.array([0.92, 0.51, 0.00, 0.00, 0.59], dtype=np.float32),  # weapon
        2: np.array([0.92, 0.53, 1.00, 0.63, 0.74], dtype=np.float32),  # camouflage
    }

    state_mean = torch.tensor(eval_dataset.state_mean, device=eval_device, dtype=torch.float32)
    state_std  = torch.tensor(eval_dataset.state_std,  device=eval_device, dtype=torch.float32)

    results = {s: [] for s in range(num_styles)}

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

            control = torch.tensor(c, dtype=torch.float32, device=device).unsqueeze(0)


            for ep in range(num_episodes_per_style):
                env = MiniGridThreeStyles(
                    target_style=STYLE_NAMES[style_id],
                    target_bonus=1.0,
                    non_target_penalty=-1.0,
                    easy_env=False,
                    agent_view_size=3,
                    randomize_layout=True,
                    **env_kwargs,
                )
                obs, _ = env.reset(seed=42 + ep)

                state = torch.from_numpy(
                    obs["image"][:, :, 0].flatten()
                ).float().to(eval_device)
                state = (state - state_mean) / state_std

                s_hist   = state.unsqueeze(0)
                a_hist   = torch.zeros(1, dtype=torch.long, device=eval_device)
                r_hist   = torch.zeros(1, device=eval_device)
                rtg_hist = torch.tensor([initial_rtg], device=eval_device)
                ti_hist  = torch.tensor([0], dtype=torch.long, device=eval_device)

                episode_return = 0.0
                done = False
                t    = 0

                while not done and t < max_ep_len:
                    action_logits = eval_model.get_action(
                        states=s_hist,
                        actions=a_hist,
                        rewards=r_hist,
                        returns_to_go=rtg_hist,
                        timesteps=ti_hist,
                        controls=control,
                    )
                    action = torch.argmax(action_logits, dim=-1).item()

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_return += float(reward)
                    t += 1

                    if not done:
                        ns = torch.from_numpy(
                            next_obs["image"][:, :, 0].flatten()
                        ).float().to(eval_device)
                        ns = (ns - state_mean) / state_std

                        s_hist   = torch.cat([s_hist,   ns.unsqueeze(0)],                                          dim=0)
                        a_hist   = torch.cat([a_hist,   torch.tensor([action],  dtype=torch.long, device=eval_device)])
                        r_hist   = torch.cat([r_hist,   torch.tensor([reward],                    device=eval_device)])
                        rtg_hist = torch.cat([rtg_hist, (rtg_hist[-1:] - reward).clamp(min=0)])
                        ti_hist  = torch.cat([ti_hist,  torch.tensor([t],       dtype=torch.long, device=eval_device)])

                results[style_id].append(episode_return)
                env.close()

            print(
                f"[control] Style {style_id} ({STYLE_NAMES[style_id]}): "
                f"mean return = {np.mean(results[style_id]):.3f}"
                f" ± {np.std(results[style_id]):.3f}"
            )

    eval_model.train()
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_eval_results(eval_history: dict, save_path: str = "plots/eval_results_control.png"):
    epochs = eval_history["epochs"]
    plt.figure(figsize=(10, 6))
    for style_id in range(3):
        returns = eval_history[f"style_{style_id}"]
        plt.plot(epochs, returns, marker="o",
                 label=f"{STYLE_NAMES[style_id].capitalize()} (Style {style_id})", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Episode Return", fontsize=12)
    plt.title("ControlDT — Online Evaluation Returns by Style", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train_control_dt(
    model:                   ControlConditionedDT,
    dataloader:              DataLoader,
    num_epochs:              int,
    device:                  str   = "cpu",
    lr:                      float = 1e-4,
    grad_clip:               float = 1.0,
    log_every:               int   = 10,
    save_path:               str   = None,
    eval_every:              int   = 10,
    eval_episodes_per_style: int   = 10,
    max_ep_len:              int   = 100,
    initial_rtg:             float = 1.0,
):
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    eval_history = {"epochs": [], "style_0": [], "style_1": [], "style_2": []}

    for epoch in range(num_epochs):
        running_loss = 0.0
        n_batches    = 0

        for batch_idx, batch in enumerate(dataloader):
            states    = batch["states"].to(device)
            actions   = batch["actions"].to(device)
            rewards   = batch["rewards"].to(device)
            rtgs      = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            controls  = batch["controls"].to(device)

            _, action_preds, _ = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtgs,
                timesteps=timesteps,
                controls=controls,
                attention_mask=attn_mask,
            )

            b, t, c = action_preds.shape
            acts_ce = actions.squeeze(-1) if actions.ndim == 3 else actions
            acts_ce = torch.clamp(acts_ce.long(), 0, c - 1)

            ce   = torch.nn.functional.cross_entropy(
                action_preds.reshape(b * t, c), acts_ce.reshape(b * t), reduction="none"
            ).reshape(b, t)
            valid = attn_mask.float()
            loss  = (ce * valid).sum() / valid.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches    += 1

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(dataloader)} "
                    f"| loss={loss.item():.6f}"
                )

        print(
            f"===> Epoch {epoch+1}/{num_epochs} "
            f"| avg_loss={running_loss/n_batches:.6f}"
        )

        if eval_every > 0 and (epoch + 1) % eval_every == 0:
            print(f"\n=== Online Evaluation at Epoch {epoch + 1} ===")
            eval_results = evaluate_online_control(
                eval_model=model,
                eval_dataset=dataloader.dataset,
                num_styles=3,
                num_episodes_per_style=eval_episodes_per_style,
                max_ep_len=max_ep_len,
                eval_device=device,
                initial_rtg=initial_rtg,
            )
            eval_history["epochs"].append(epoch + 1)
            for style_id in range(3):
                mean_r = np.mean(eval_results[style_id]) if eval_results[style_id] else 0.0
                eval_history[f"style_{style_id}"].append(mean_r)
            print()

        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    if eval_history["epochs"]:
        plot_eval_results(eval_history)

    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    max_len = 8
    dataset_params = {
        "sampling":                   True,
        "index_channel_only":         True,
        "state_normalization_factor":  1,
        "action_normalization_factor": 1,
        "max_len":                    max_len,
        "control_dim":                CONTROL_DIM,
    }
    dataset = ControlDataset(trajectory_paths=paths, **dataset_params)
    loader  = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn
    )

    model = ControlConditionedDT(
        state_dim=9,
        act_dim=7,
        hidden_size=128,
        control_dim=CONTROL_DIM,
        max_length=max_len,
        max_ep_len=100,
        action_tanh=False,
        n_layer=4,
        n_head=8,
    )

    train_control_dt(
        model=model,
        dataloader=loader,
        num_epochs=100,
        device=device,
        lr=1e-3,
        grad_clip=1.0,
        log_every=10,
        save_path="trained_models/control_dt_minigrid.pth",
    )
