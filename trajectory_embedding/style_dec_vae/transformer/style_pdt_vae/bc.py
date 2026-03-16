"""
Behavioural Cloning (BC) oracle baseline for the MiniGrid three-style env.

Trains one independent MLP policy per style using only that style's
trajectories.  This is an "oracle" baseline — it has privileged access to the
ground-truth style label at training time, which the other methods do not.

At evaluation, the correct policy is simply selected for the desired style,
giving the best achievable performance with full style supervision and no
latent structure required.

Mirrors prompt_dt.py / control_prompt_pdt.py in dataset, evaluation, and
__main__ structure for direct comparison.
"""

from typing import Callable, Dict, Optional, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from envs.three_style_env import MiniGridThreeStyles
from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths


STYLE_NAMES = {0: "bypass", 1: "weapon", 2: "camouflage"}


# =============================================================================
# Dataset
# =============================================================================

class BCDataset(Dataset):
    """
    Flat (state, action) pairs drawn from trajectories of a single style.

    Every timestep from every trajectory of the target style becomes one
    independent training sample — no temporal context.
    """

    def __init__(
        self,
        trajectory_paths,
        target_style:               int,
        index_channel_only:         bool  = False,
        state_normalization_factor: float = 1,
        action_normalization_factor:float = 1,
        pct_traj:                   float = 1.0,
    ):
        self.target_style = target_style

        # Use TrajectoryDataset to handle all loading, infos, filtering, norms.
        base = TrajectoryDataset(
            trajectory_paths=trajectory_paths,
            max_len=1,          # window length irrelevant — we extract raw steps
            sampling=True,
            index_channel_only=index_channel_only,
            state_normalization_factor=state_normalization_factor,
            action_normalization_factor=action_normalization_factor,
            pct_traj=pct_traj,
        )

        self.state_mean = base.state_mean   # [state_dim_flat]
        self.state_std  = base.state_std    # [state_dim_flat]

        states_list  = []
        actions_list = []
        n_trajs      = 0

        for traj_i in base.indices:
            if int(base.tasks[traj_i]) != target_style:
                continue

            s_raw = base.states[traj_i]   # [T, *state_dim]
            a_raw = base.actions[traj_i]  # [T, *act_dim]

            # normalise to numpy
            if isinstance(s_raw, torch.Tensor):
                s_raw = s_raw.numpy()
            else:
                s_raw = np.asarray(s_raw)
            if isinstance(a_raw, torch.Tensor):
                a_raw = a_raw.numpy()
            else:
                a_raw = np.asarray(a_raw)

            # flatten spatial dims → [T, state_dim_flat]
            s_norm = (s_raw.reshape(len(s_raw), -1) - self.state_mean) / self.state_std
            states_list.append(s_norm.astype(np.float32))
            actions_list.append(a_raw.reshape(len(a_raw)).astype(np.int64))
            n_trajs += 1

        if not states_list:
            raise ValueError(
                f"No trajectories found for style {target_style} "
                f"({STYLE_NAMES.get(target_style, '?')})."
            )

        self.states  = np.concatenate(states_list,  axis=0)  # [N, state_dim_flat]
        self.actions = np.concatenate(actions_list, axis=0)  # [N]

        print(
            f"  Style {target_style} ({STYLE_NAMES[target_style]}): "
            f"{len(self.states)} steps from {n_trajs} trajectories"
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
        )


# =============================================================================
# Policy
# =============================================================================

class BCPolicy(nn.Module):
    """
    Simple MLP policy: state → action logits.

    No temporal context — a pure memoryless reactive policy trained with
    cross-entropy loss.
    """

    def __init__(
        self,
        state_dim:   int,
        act_dim:     int,
        hidden_size: int = 256,
        num_layers:  int = 3,
    ):
        super().__init__()

        dims   = [state_dim] + [hidden_size] * num_layers + [act_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [B, state_dim] → logits: [B, act_dim]"""
        return self.net(states)

    def get_action(self, state: torch.Tensor) -> int:
        """state: [state_dim] → greedy action (int)"""
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())


# =============================================================================
# Training
# =============================================================================

def train_bc_style(
    style_id:                    int,
    trajectory_paths,
    act_dim:                     int   = 7,
    hidden_size:                 int   = 256,
    num_layers:                  int   = 3,
    batch_size:                  int   = 256,
    num_epochs:                  int   = 100,
    lr:                          float = 1e-3,
    weight_decay:                float = 1e-4,
    device:                      str   = "cpu",
    log_every:                   int   = 10,
    save_path:                   str   = None,
    index_channel_only:          bool  = True,
    state_normalization_factor:  float = 1,
    action_normalization_factor: float = 1,
    pct_traj:                    float = 1.0,
) -> Tuple[BCPolicy, BCDataset]:
    """
    Train a BC policy for one style.  Returns (policy, dataset).
    The dataset carries state_mean / state_std needed at eval time.
    """
    dataset = BCDataset(
        trajectory_paths=trajectory_paths,
        target_style=style_id,
        index_channel_only=index_channel_only,
        state_normalization_factor=state_normalization_factor,
        action_normalization_factor=action_normalization_factor,
        pct_traj=pct_traj,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    state_dim_flat = dataset.states.shape[1]
    policy = BCPolicy(
        state_dim=state_dim_flat,
        act_dim=act_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    policy.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        n_batches    = 0

        for states, actions in loader:
            states  = states.to(device)
            actions = actions.to(device)

            loss = torch.nn.functional.cross_entropy(policy(states), actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches    += 1

        if log_every > 0 and (epoch + 1) % log_every == 0:
            print(
                f"  [{STYLE_NAMES[style_id]}] "
                f"Epoch {epoch+1}/{num_epochs} "
                f"| avg_loss={running_loss/n_batches:.4f}"
            )

    if save_path is not None:
        torch.save(policy.state_dict(), save_path)
        print(f"  Saved {save_path}")

    return policy, dataset


def train_all_styles(
    trajectory_paths,
    act_dim:     int   = 7,
    hidden_size: int   = 256,
    num_layers:  int   = 3,
    batch_size:  int   = 256,
    num_epochs:  int   = 100,
    lr:          float = 1e-3,
    device:      str   = "cpu",
    log_every:   int   = 10,
    save_dir:    str   = ".",
    index_channel_only:          bool  = True,
    state_normalization_factor:  float = 1,
    action_normalization_factor: float = 1,
) -> Dict[int, Tuple[BCPolicy, BCDataset]]:
    """
    Train one BC policy per style.

    Returns:
        policies: {style_id: (policy, dataset)}
    """
    policies = {}
    for style_id in range(3):
        print(f"\n=== Training BC for Style {style_id} ({STYLE_NAMES[style_id]}) ===")
        save_path = os.path.join(save_dir, f"trained_models/bc_style{style_id}.pth")
        policy, dataset = train_bc_style(
            style_id=style_id,
            trajectory_paths=trajectory_paths,
            act_dim=act_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            log_every=log_every,
            save_path=save_path,
            index_channel_only=index_channel_only,
            state_normalization_factor=state_normalization_factor,
            action_normalization_factor=action_normalization_factor,
        )
        policies[style_id] = (policy, dataset)
    return policies


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_bc_oracle(
    policies:               Dict[int, Tuple[BCPolicy, BCDataset]],
    num_episodes_per_style: int   = 10,
    max_ep_len:             int   = 100,
    eval_device:            str   = "cpu",
    initial_rtg:            float = 1.0,   # unused; kept for API symmetry
    env_kwargs:             dict  = None,
) -> Dict[int, list]:
    """
    Roll out each style's dedicated BC policy on its own style's environment.

    Returns:
        results: {style_id: [episode_returns]}
    """
    if env_kwargs is None:
        env_kwargs = {}

    results = {s: [] for s in policies}

    for style_id, (policy, dataset) in policies.items():
        policy.eval()
        state_mean = torch.tensor(dataset.state_mean, dtype=torch.float32, device=eval_device)
        state_std  = torch.tensor(dataset.state_std,  dtype=torch.float32, device=eval_device)

        with torch.no_grad():
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

                episode_return = 0.0
                done = False
                t    = 0

                while not done and t < max_ep_len:
                    state = torch.from_numpy(
                        obs["image"][:, :, 0].flatten()
                    ).float().to(eval_device)
                    state = (state - state_mean) / state_std

                    action = policy.get_action(state)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_return += float(reward)
                    t += 1
                results[style_id].append(episode_return)
                env.close()

        print(
            f"[BC oracle] Style {style_id} ({STYLE_NAMES[style_id]}): "
            f"mean return = {np.mean(results[style_id]):.3f}"
            f" ± {np.std(results[style_id]):.3f}"
        )

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_eval_results(results: Dict[int, list], save_path: str = "plots/eval_results_bc.png"):
    """Bar chart of mean ± std return per style."""
    styles  = sorted(results.keys())
    means   = [np.mean(results[s]) for s in styles]
    stds    = [np.std(results[s])  for s in styles]
    labels  = [STYLE_NAMES[s].capitalize() for s in styles]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors[:len(styles)],
           alpha=0.85, error_kw={"linewidth": 1.5})
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("BC Oracle — Return by Style", fontsize=14)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    policies = train_all_styles(
        trajectory_paths=paths,
        act_dim=7,
        hidden_size=256,
        num_layers=3,
        batch_size=256,
        num_epochs=100,
        lr=1e-3,
        device=device,
        log_every=10,
        save_dir=".",
        index_channel_only=True,
        state_normalization_factor=1,
        action_normalization_factor=1,
    )

    print("\n=== Evaluation ===")
    results = evaluate_bc_oracle(
        policies=policies,
        num_episodes_per_style=20,
        max_ep_len=100,
        eval_device=device,
    )
    plot_eval_results(results)

    # --- Load checkpoints and evaluate ---
    # policies = {}
    # for style_id in range(3):
    #     ckpt = os.path.join("trained_models", f"bc_style{style_id}.pth")
    #     dataset = BCDataset(
    #         trajectory_paths=paths,
    #         target_style=style_id,
    #         index_channel_only=True,
    #         state_normalization_factor=1,
    #         action_normalization_factor=1,
    #     )
    #     policy = BCPolicy(state_dim=dataset.states.shape[1], act_dim=7, hidden_size=256, num_layers=3)
    #     policy.load_state_dict(torch.load(ckpt, map_location=device))
    #     policy.to(device)
    #     print(f"Loaded {ckpt}")
    #     policies[style_id] = (policy, dataset)
    #
    # print("\n=== Evaluation ===")
    # results = evaluate_bc_oracle(
    #     policies=policies,
    #     num_episodes_per_style=20,
    #     max_ep_len=100,
    #     eval_device=device,
    # )
    # plot_eval_results(results)
