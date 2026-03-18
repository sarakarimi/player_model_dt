"""
Prompting Decision Transformer (PromptDT) for the MiniGrid three-style env.

A full reference trajectory (the "prompt") is prepended to the current context
window before being fed to the transformer.  The model attends to both and is
trained with a standard BC loss on the current-window actions only.

At inference, pick any trajectory from the dataset for the desired style and
pass it as the prompt — no encoder or learned latent required.

Mirrors pdt_vae_with_prior.py exactly in backbone, dataset, training loop, and
evaluation structure so the two can be compared directly.
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


# =============================================================================
# Dataset
# =============================================================================

class PromptDataset(TrajectoryDataset):
    """
    Returns a current-context window AND a fixed-length prompt trajectory
    sampled from the same style.

    Prompt sampling follows prompt_utils.get_prompt() from the PromptDT paper:
      - Takes the last `prompt_length` steps of the reference trajectory
        (si = max(0, traj_len - prompt_length)), so the prompt always has a
        fixed, well-defined length rather than varying by trajectory.
    Batch item:
        s, a, r, d, rtg, ti, m            – current window
                                             length: max_len
        p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m  – prompt
                                             length: prompt_length
        task_label
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
        prompt_length: int = None,        # steps per prompt episode; defaults to max_len
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

        self.prompt_length = prompt_length

        # Pre-build style → index lookup so prompt sampling is O(1)
        self.style_to_indices: dict = {}
        for idx in self.indices:
            label = int(self.tasks[idx])
            self.style_to_indices.setdefault(label, []).append(idx)

    # ------------------------------------------------------------------
    def get_prompt_traj(self, traj_index: int):
        """
        Returns the last self.prompt_length steps of the trajectory,
        padded to exactly self.prompt_length.

        Matches prompt_utils.get_prompt() from the PromptDT paper:
            si = max(0, traj_len - prompt_length)
        RTG is computed as cumulative sum from right (returns-to-go).
        """
        traj_rewards = self.rewards[traj_index]
        traj_states  = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones   = self.dones[traj_index]
        traj_len     = traj_rewards.shape[0]

        # Random start in [0, traj_len - prompt_length] so the slice always has
        # prompt_length steps and the last prompt_length steps can be selected.
        si  = random.randint(0, max(0, traj_len - self.prompt_length))
        end = si + self.prompt_length

        s   = traj_states[si:end].reshape(1, -1, *self.state_dim)
        a   = traj_actions[si:end].reshape(1, -1, *self.act_dim)
        r   = traj_rewards[si:end].reshape(1, -1, 1)
        d   = traj_dones[si:end].reshape(1, -1)
        ti  = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # RTG: cumulative sum of rewards from each step to end of trajectory.
        # Convert to numpy first — PyTorch tensors don't support [::-1] slicing.
        rewards_np = traj_rewards.numpy() if isinstance(traj_rewards, torch.Tensor) else np.asarray(traj_rewards)
        rtg_vals = np.cumsum(rewards_np[si:][::-1])[::-1]
        rtg = rtg_vals[:s.shape[1]].reshape(1, -1, 1)

        actual_len = s.shape[1]
        padding    = self.prompt_length - actual_len

        s   = self.add_padding(s,   0,         padding)
        a   = self.add_padding(a,   -10,        padding)
        r   = self.add_padding(r,   0,          padding)
        d   = self.add_padding(d,   2,          padding)
        ti  = self.add_padding(ti,  0,          padding)
        rtg = self.add_padding(rtg, 0,          padding)
        m   = self.add_padding(np.ones((1, actual_len)), 0, padding)

        s   = (s   - self.state_mean) / self.state_std
        rtg = rtg  / self.rtg_scale

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(torch.from_numpy(s).float())
            s = s.numpy()

        return self.return_tensors(s, a, r, rtg, d, ti, m)

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
        s   = self.add_padding(s,   0,    padding)
        a   = self.add_padding(a,   -10,  padding)
        r   = self.add_padding(r,   0,    padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d   = self.add_padding(d,   2,    padding)
        ti  = self.add_padding(ti,  0,    padding)
        m   = self.add_padding(np.ones((1, tlen)), 0, padding)

        s   = (s   - self.state_mean) / self.state_std
        rtg = rtg  / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        traj_index = self.indices[idx]
        task_label = int(self.tasks[traj_index])

        # current context window
        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end,
        )

        candidates = self.style_to_indices.get(task_label, [traj_index])
        other = [i for i in candidates if i != traj_index]

        p_idx = random.choice(other) if other else traj_index
        p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m = self.get_prompt_traj(p_idx)

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)

        return s, a, r, d, rtg, ti, m, p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m, task_label

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        (
            states, actions, rewards, dones, rtgs, timesteps, masks,
            p_states, p_actions, p_rewards, p_dones, p_rtgs, p_timesteps, p_masks,
            task_labels,
        ) = zip(*batch)

        return {
            "states":           torch.stack(states,    dim=0),
            "actions":          torch.stack(actions,   dim=0),
            "rewards":          torch.stack(rewards,   dim=0),
            "returns_to_go":    torch.stack(rtgs,      dim=0),
            "timesteps":        torch.stack(timesteps, dim=0),
            "attention_mask":   torch.stack(masks,     dim=0),
            "dones":            torch.stack(dones,     dim=0),
            # prompt
            "prompt_states":        torch.stack(p_states,    dim=0),
            "prompt_actions":       torch.stack(p_actions,   dim=0),
            "prompt_rewards":       torch.stack(p_rewards,   dim=0),
            "prompt_returns_to_go": torch.stack(p_rtgs,      dim=0),
            "prompt_timesteps":     torch.stack(p_timesteps, dim=0),
            "prompt_attention_mask":torch.stack(p_masks,     dim=0),
            "prompt_dones":         torch.stack(p_dones,     dim=0),
            # label
            "task_labels": torch.tensor(task_labels, dtype=torch.long),
        }


# =============================================================================
# Prompting Decision Transformer
# =============================================================================

class PromptingDecisionTransformer(nn.Module):
    """
    Decision Transformer with trajectory prompting (PromptDT).

    A reference trajectory (prompt) is prepended to the current context as
    (rtg, state, action) token triples before being fed to the transformer.
    Separate embedding weights are used for prompt and current-context tokens,
    following the PromptDT paper.  LayerNorm is applied to the current-context
    tokens only, matching the reference implementation.

    prompt is passed as a 7-tuple:
        (states, actions, rewards, dones, returns_to_go, timesteps, attention_mask)

    When prompt is None the model behaves as a vanilla Decision Transformer.

    Note: actions are discrete integers → nn.Embedding is used instead of
    nn.Linear (MiniGrid has 7 discrete actions).
    """

    def __init__(
        self,
        state_dim: int,
        act_dim:   int,
        hidden_size: int,
        max_length:  Optional[int] = None,
        max_ep_len:  int = 4096,
        action_tanh: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size
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

        # Separate prompt embeddings (independent weights, following PromptDT)
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return   = nn.Linear(1, hidden_size)
        self.prompt_embed_state    = nn.Linear(state_dim, hidden_size)
        self.prompt_embed_action   = nn.Embedding(act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state  = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        prompt=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        # ------ embed current context ----------------------------------------
        acts = torch.clamp(actions.long().squeeze(-1), 0, self.act_dim - 1)
        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        time_embeddings    = self.embed_timestep(timesteps)
        state_embeddings   = self.embed_state(states)   + time_embeddings
        action_embeddings  = self.embed_action(acts)    + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)   # LN on current context only

        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # ------ embed & prepend prompt ---------------------------------------
        if prompt is not None:
            (
                p_states, p_actions, p_rewards, p_dones,
                p_returns_to_go, p_timesteps, p_attention_mask,
            ) = prompt
            prompt_seq_length = p_states.shape[1]

            p_acts = torch.clamp(p_actions.long().squeeze(-1), 0, self.act_dim - 1)
            if p_returns_to_go.ndim == 2:
                p_returns_to_go = p_returns_to_go.unsqueeze(-1)

            p_time_embeddings    = self.prompt_embed_timestep(p_timesteps)
            p_state_embeddings   = self.prompt_embed_state(p_states)   + p_time_embeddings
            p_action_embeddings  = self.prompt_embed_action(p_acts)    + p_time_embeddings
            p_returns_embeddings = self.prompt_embed_return(p_returns_to_go) + p_time_embeddings

            prompt_stacked_inputs = (
                torch.stack(
                    (p_returns_embeddings, p_state_embeddings, p_action_embeddings), dim=1
                )
                .permute(0, 2, 1, 3)
                .reshape(p_states.shape[0], 3 * prompt_seq_length, self.hidden_size)
            )
            prompt_stacked_attention_mask = (
                torch.stack(
                    (p_attention_mask, p_attention_mask, p_attention_mask), dim=1
                )
                .permute(0, 2, 1)
                .reshape(p_states.shape[0], 3 * prompt_seq_length)
            )

            # broadcast a single shared prompt to the full batch
            if prompt_stacked_inputs.shape[0] != batch_size:
                prompt_stacked_inputs         = prompt_stacked_inputs.expand(batch_size, -1, -1)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.expand(batch_size, -1)

            stacked_inputs         = torch.cat([prompt_stacked_inputs,         stacked_inputs],         dim=1)
            stacked_attention_mask = torch.cat([prompt_stacked_attention_mask, stacked_attention_mask], dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        if prompt is None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # note: prompt tokens are pre-pended; slice the last seq_length steps only
        return_preds = self.predict_return(x[:, 2])[:, -seq_length:, :]
        state_preds  = self.predict_state(x[:, 2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:, 1])[:, -seq_length:, :]

        return state_preds, action_preds, return_preds

    # ------------------------------------------------------------------
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """
        Inference helper: pads/trims context to max_length, returns the
        predicted action logits for the last timestep.

        Pass prompt as a keyword argument, e.g.:
            model.get_action(..., prompt=(p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m))
        """
        states        = states.reshape(1, -1, self.state_dim)
        actions       = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps     = timesteps.reshape(1, -1)

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
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, **kwargs
        )
        return action_preds[0, -1]


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_online_prompting(
    eval_model:             PromptingDecisionTransformer,
    eval_dataset:           PromptDataset,
    num_styles:             int   = 3,
    num_episodes_per_style: int   = 10,
    max_ep_len:             int   = 100,
    eval_device:            str   = "cpu",
    initial_rtg:            float = 1.0,
    env_kwargs:             dict  = None,
):
    """
    For each style, sample one representative prompt trajectory from the dataset
    and roll out the model conditioned on it via get_action.

    Returns:
        results: {style_id: [episode_returns]}
    """
    eval_model.eval()
    if env_kwargs is None:
        env_kwargs = {}

    style_names = {0: "bypass", 1: "weapon", 2: "camouflage"}

    state_mean = torch.tensor(eval_dataset.state_mean, device=eval_device, dtype=torch.float32)
    state_std  = torch.tensor(eval_dataset.state_std,  device=eval_device, dtype=torch.float32)

    results = {s: [] for s in range(num_styles)}

    with torch.no_grad():
        for style_id in range(num_styles):
            candidates = eval_dataset.style_to_indices.get(style_id, [])
            if not candidates:
                print(f"  No trajectories for style {style_id}, skipping.")
                continue


            prompt_idx = random.choice(candidates)
            p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m = eval_dataset.get_prompt_traj(prompt_idx)

            # add batch dim and move to device
            p_s   = p_s.unsqueeze(0).to(eval_device)
            p_a   = p_a.unsqueeze(0).to(eval_device)
            p_r   = p_r.unsqueeze(0).to(eval_device)
            p_d   = p_d.unsqueeze(0).to(eval_device)
            p_rtg = p_rtg.unsqueeze(0).to(eval_device)
            p_ti  = p_ti.unsqueeze(0).to(eval_device)
            p_m   = p_m.unsqueeze(0).float().to(eval_device)
            prompt_tuple = (p_s, p_a, p_r, p_d, p_rtg, p_ti, p_m)

            for ep in range(num_episodes_per_style):
                env = MiniGridThreeStyles(
                    target_style=style_names[style_id],
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

                # history buffers (grow each step; get_action handles windowing)
                s_hist  = state.unsqueeze(0)                                      # [1, state_dim]
                a_hist  = torch.zeros(1, dtype=torch.long, device=eval_device)    # [1]
                r_hist  = torch.zeros(1, device=eval_device)                      # [1]
                rtg_hist = torch.tensor([initial_rtg], device=eval_device)        # [1]
                ti_hist  = torch.tensor([0], dtype=torch.long, device=eval_device)# [1]

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
                        prompt=prompt_tuple,
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

                        s_hist   = torch.cat([s_hist,   ns.unsqueeze(0)],                                        dim=0)
                        a_hist   = torch.cat([a_hist,   torch.tensor([action], dtype=torch.long, device=eval_device)])
                        r_hist   = torch.cat([r_hist,   torch.tensor([reward], device=eval_device)])
                        rtg_hist = torch.cat([rtg_hist, (rtg_hist[-1:] - reward).clamp(min=0)])
                        ti_hist  = torch.cat([ti_hist,  torch.tensor([t], dtype=torch.long, device=eval_device)])

                results[style_id].append(episode_return)
                env.close()

            print(
                f"[prompt] Style {style_id} ({style_names[style_id]}): "
                f"mean return = {np.mean(results[style_id]):.3f}"
                f" ± {np.std(results[style_id]):.3f}"
            )

    eval_model.train()
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_eval_results(eval_history: dict, save_path: str = "plots/eval_results_prompt.png"):
    epochs      = eval_history["epochs"]
    style_names = {0: "Bypass", 1: "Weapon", 2: "Camouflage"}

    plt.figure(figsize=(10, 6))
    for style_id in range(3):
        returns = eval_history[f"style_{style_id}"]
        plt.plot(epochs, returns, marker="o",
                 label=f"{style_names[style_id]} (Style {style_id})", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Episode Return", fontsize=12)
    plt.title("PromptDT — Online Evaluation Returns by Style", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()


# =============================================================================
# Training
# =============================================================================

def train_prompting_dt(
    model:               PromptingDecisionTransformer,
    dataloader:          DataLoader,
    num_epochs:          int,
    device:              str   = "cpu",
    lr:                  float = 1e-4,
    grad_clip:           float = 1.0,
    log_every:           int   = 10,
    save_path:           str   = None,
    eval_every:          int   = 10,
    eval_episodes_per_style: int = 10,
    max_ep_len:          int   = 100,
    initial_rtg:         float = 1.0,
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

            p_states    = batch["prompt_states"].to(device)
            p_actions   = batch["prompt_actions"].to(device)
            p_rewards   = batch["prompt_rewards"].to(device)
            p_dones     = batch["prompt_dones"].to(device)
            p_rtgs      = batch["prompt_returns_to_go"].to(device)
            p_timesteps = batch["prompt_timesteps"].to(device)
            p_mask      = batch["prompt_attention_mask"].float().to(device)

            prompt = (p_states, p_actions, p_rewards, p_dones, p_rtgs, p_timesteps, p_mask)

            _, action_preds, _ = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
                prompt=prompt,
            )

            b, t, c = action_preds.shape
            acts_ce = actions.squeeze(-1) if actions.ndim == 3 else actions
            acts_ce = torch.clamp(acts_ce.long(), 0, c - 1)

            logits  = action_preds.reshape(b * t, c)
            targets = acts_ce.reshape(b * t)

            ce    = torch.nn.functional.cross_entropy(logits, targets, reduction="none").reshape(b, t)
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
            eval_results = evaluate_online_prompting(
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
        "prompt_length":              2,   # steps per prompt episode (last N steps)
    }
    dataset = PromptDataset(trajectory_paths=paths, **dataset_params)
    loader  = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn
    )

    model = PromptingDecisionTransformer(
        state_dim=9,
        act_dim=7,
        hidden_size=128,
        max_length=max_len,
        max_ep_len=100,
        action_tanh=False,
        n_layer=4,
        n_head=8,
    )

    train_prompting_dt(
        model=model,
        dataloader=loader,
        num_epochs=100,
        device=device,
        lr=1e-3,
        grad_clip=1.0,
        log_every=10,
        save_path="trained_models/prompt_dt_minigrid.pth",
    )