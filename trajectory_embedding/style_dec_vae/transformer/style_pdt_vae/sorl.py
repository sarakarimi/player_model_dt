"""
SORL: Style-conditioned Offline RL baseline (ICLR 2024).

Adapted from https://github.com/cedesu/SORL for the MiniGrid three-style env.

SORL discovers K behavioral styles from *unlabeled* offline data via EM:
  Stage 1 — Value network: train V(s) to predict trajectory returns (MSE).
  Stage 2 — EM training:
    E-step: soft-assign each trajectory to K styles via BC log-likelihoods.
    M-step: advantage-weighted BC per style using the soft assignments.

Backbone: K independent BC MLP policies (state → action logits), one per
discovered style.  This is faithful to the original SORL paper.  Unlike the
supervised BC baseline in bc.py, SORL does NOT use ground-truth style labels —
it discovers styles automatically from the mixed offline dataset.

Usage (standalone):
    python sorl.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from envs.three_style_env import MiniGridThreeStyles
from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_STYLES  = 3
STYLE_NAMES = {0: "bypass", 1: "weapon", 2: "camouflage"}


# =============================================================================
# Dataset
# =============================================================================

class SODataset(Dataset):
    """
    Mixed-style flat dataset for SORL.

    Loads trajectories of all styles without style labels (unsupervised).
    Exposes two views of the data:

    Trajectory-level (for E-step and value network):
      traj_states[i]  — np.ndarray [T_i, state_dim_flat], normalised
      traj_actions[i] — np.ndarray [T_i], int actions
      traj_returns[i] — float, total episode return

    Flat (for M-step):
      flat_states    — np.ndarray [N_steps, state_dim_flat]
      flat_actions   — np.ndarray [N_steps], int actions
      flat_traj_ids  — np.ndarray [N_steps], dataset trajectory index (0…N-1)

    __getitem__ returns flat (state, action, traj_id) triples for M-step.
    """

    def __init__(
        self,
        trajectory_paths,
        sampling:                    bool  = True,
        index_channel_only:          bool  = True,
        state_normalization_factor:  float = 1,
        action_normalization_factor: float = 1,
        pct_traj:                    float = 1.0,
    ):
        base = TrajectoryDataset(
            trajectory_paths=trajectory_paths,
            max_len=1,                           # window length irrelevant here
            sampling=sampling,
            index_channel_only=index_channel_only,
            state_normalization_factor=state_normalization_factor,
            action_normalization_factor=action_normalization_factor,
            pct_traj=pct_traj,
            normalize_state=True,
        )

        self.state_mean      = base.state_mean   # [state_dim_flat]
        self.state_std       = base.state_std    # [state_dim_flat]
        self.state_dim_flat  = int(np.prod(list(base.states[0][0].shape)))

        # Build trajectory-level and flat views, iterating only over base.indices
        traj_states_list  = []
        traj_actions_list = []
        traj_returns_list = []
        flat_s_list       = []
        flat_a_list       = []
        flat_tid_list     = []

        for dataset_i, traj_i in enumerate(base.indices):
            s_raw = base.states[traj_i]
            a_raw = base.actions[traj_i]

            if isinstance(s_raw, torch.Tensor):
                s_raw = s_raw.numpy()
            else:
                s_raw = np.asarray(s_raw)
            if isinstance(a_raw, torch.Tensor):
                a_raw = a_raw.numpy()
            else:
                a_raw = np.asarray(a_raw)

            s_norm = (
                (s_raw.reshape(len(s_raw), -1) - self.state_mean) / self.state_std
            ).astype(np.float32)
            a_flat = a_raw.reshape(len(a_raw)).astype(np.int64)
            total_return = float(base.rewards[traj_i].sum())

            traj_states_list.append(s_norm)
            traj_actions_list.append(a_flat)
            traj_returns_list.append(total_return)

            flat_s_list.append(s_norm)
            flat_a_list.append(a_flat)
            flat_tid_list.append(
                np.full(len(s_norm), dataset_i, dtype=np.int64)
            )

        self.traj_states  = traj_states_list
        self.traj_actions = traj_actions_list
        self.traj_returns = np.array(traj_returns_list, dtype=np.float32)
        self.num_trajectories = len(traj_states_list)

        self.flat_states   = np.concatenate(flat_s_list,  axis=0)   # [N_steps, S]
        self.flat_actions  = np.concatenate(flat_a_list,  axis=0)   # [N_steps]
        self.flat_traj_ids = np.concatenate(flat_tid_list, axis=0)  # [N_steps]

        print(
            f"SODataset: {self.num_trajectories} trajectories, "
            f"{len(self.flat_states)} timesteps, "
            f"state_dim={self.state_dim_flat}"
        )

    def __len__(self) -> int:
        return len(self.flat_states)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.flat_states[idx]),
            torch.tensor(self.flat_actions[idx], dtype=torch.long),
            torch.tensor(self.flat_traj_ids[idx], dtype=torch.long),
        )


# =============================================================================
# Value Network  (Stage 1)
# =============================================================================

class ValueNetwork(nn.Module):
    """
    MLP V(s) trained to predict total trajectory return from the first state.
    Used for advantage estimation A_i = R_i - V(s_{i,0}) in the SORL M-step.
    """

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [B, state_dim] → values: [B]"""
        return self.net(states).squeeze(-1)


def train_value_network(
    value_net:  ValueNetwork,
    dataset:    SODataset,
    num_epochs: int   = 20,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    device:     str   = "cpu",
    save_path:  str   = None,
) -> ValueNetwork:
    """Train V(s) via MSE on (first_state, total_return) pairs."""
    value_net.to(device)
    value_net.train()
    optimizer = AdamW(value_net.parameters(), lr=lr)

    first_states = torch.from_numpy(
        np.stack([s[0] for s in dataset.traj_states])
    ).float()                                              # [N, S]
    returns_t = torch.tensor(dataset.traj_returns, dtype=torch.float32)

    N       = len(first_states)
    idx_all = np.arange(N)

    for epoch in range(num_epochs):
        np.random.shuffle(idx_all)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            ib  = idx_all[start:start + batch_size]
            s_b = first_states[ib].to(device)
            r_b = returns_t[ib].to(device)

            loss = F.mse_loss(value_net(s_b), r_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        print(
            f"[Value] Epoch {epoch+1:3d}/{num_epochs}  "
            f"loss={total_loss / n_batches:.4f}"
        )

    if save_path is not None:
        torch.save(value_net.state_dict(), save_path)
        print(f"Saved value network → {save_path}")

    return value_net


# =============================================================================
# BC Policy (per-style backbone)
# =============================================================================

class BCPolicy(nn.Module):
    """
    Simple MLP: state → action logits.  Same architecture as bc.py.
    No temporal context — memoryless reactive policy.
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
        """state: [state_dim] → greedy action int"""
        with torch.no_grad():
            return int(torch.argmax(self.forward(state.unsqueeze(0)), dim=-1).item())


# =============================================================================
# SORL E-Step: Soft Trajectory Assignments
# =============================================================================

@torch.no_grad()
def compute_soft_assignments(
    policies:    list,
    dataset:     SODataset,
    device:      str   = "cpu",
    temperature: float = 1.0,
    chunk:       int   = 4096,
) -> np.ndarray:
    """
    E-step: compute soft assignment W[N, K].

    For each trajectory i and style k:
        log_like[i, k] = Σ_t  log π_k(a_t | s_t)
    W[i, :] = softmax(log_like[i, :] / temperature)

    Uses the flat view of the dataset for efficient batch inference, then
    scatter-sums per-timestep log-probs back to trajectory level.

    Returns W: np.ndarray [N_trajs, K], float32.
    """
    K = len(policies)
    N = dataset.num_trajectories

    all_s = torch.from_numpy(dataset.flat_states)    # [N_steps, S]
    all_a = torch.from_numpy(dataset.flat_actions)   # [N_steps]  int
    tids  = dataset.flat_traj_ids                    # [N_steps]  int, numpy

    log_likes = np.zeros((N, K), dtype=np.float64)

    for k, policy in enumerate(policies):
        policy.eval()
        per_step_lp = np.empty(len(all_s), dtype=np.float32)

        for start in range(0, len(all_s), chunk):
            s_b   = all_s[start:start + chunk].to(device)
            a_b   = all_a[start:start + chunk].to(device)
            logits = policy(s_b)
            lp     = F.log_softmax(logits, dim=-1)           # [B, act_dim]
            step_lp = lp[torch.arange(len(a_b)), a_b]       # [B]
            per_step_lp[start:start + chunk] = step_lp.cpu().numpy()

        # scatter-sum: log_likes[traj_i, k] += per_step_lp[t] for all t in traj_i
        np.add.at(log_likes[:, k], tids, per_step_lp)

    # softmax over K with temperature
    log_likes /= max(temperature, 1e-8)
    log_likes -= log_likes.max(axis=1, keepdims=True)   # numerical stability
    W = np.exp(log_likes)
    W /= W.sum(axis=1, keepdims=True)

    for policy in policies:
        policy.train()

    return W.astype(np.float32)                          # [N, K]


# =============================================================================
# SORL Training (EM loop)
# =============================================================================

def train_sorl(
    policies:         list,
    value_net:        ValueNetwork,
    dataset:          SODataset,
    num_em_iters:     int   = 10,
    m_step_epochs:    int   = 5,
    batch_size:       int   = 256,
    lr:               float = 1e-3,
    grad_clip:        float = 1.0,
    device:           str   = "cpu",
    beta:             float = 1.0,
    adv_clip:         float = 5.0,
    temperature:      float = 1.0,
    log_every:        int   = 50,
    save_dir:         str   = "trained_models",
    warmup_bc_epochs: int   = 5,
    eval_every:       int   = 2,
    num_eval_ep:      int   = 10,
    initial_rtg:      float = 1.0,
    max_ep_len:       int   = 100,
) -> list:
    """
    Full SORL training: optional warm-up BC then EM loop.

    Warm-up: uniform-weight BC across all K styles (equal W[i,k] = 1/K).
    EM:
      E-step  — W[i, k] = softmax(Σ_t log π_k(a_t|s_t) / τ)
      M-step  — minimise per-policy loss:
                  L_k = mean_i  W[i,k] · exp(clip(A_i/β, ±adv_clip)) · CE(π_k, a_i)

    policies: list of K BCPolicy instances (one per discovered style).
    """
    K = len(policies)
    for p in policies:
        p.to(device)
        p.train()
    value_net.to(device)
    value_net.eval()

    optimizers = [AdamW(p.parameters(), lr=lr) for p in policies]

    # Precompute per-trajectory advantages: A_i = R_i - V(s_{i,0})
    print("Computing per-trajectory advantages …")
    first_states = torch.from_numpy(
        np.stack([s[0] for s in dataset.traj_states])
    ).float()                                              # [N, S]

    with torch.no_grad():
        chunk = 512
        Vs = []
        for start in range(0, len(first_states), chunk):
            Vs.append(value_net(first_states[start:start + chunk].to(device)).cpu())
        Vs = torch.cat(Vs).numpy()                        # [N]

    traj_returns = dataset.traj_returns                   # [N]  numpy
    advantages   = (traj_returns - Vs).astype(np.float32) # [N]

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    # ── Warm-up BC (uniform W = 1/K) ─────────────────────────────────────────
    if warmup_bc_epochs > 0:
        print(f"==> Warm-up BC for {warmup_bc_epochs} epoch(s)")
        for ep in range(warmup_bc_epochs):
            total_losses = [0.0] * K
            for bidx, (states, actions, _) in enumerate(loader):
                states  = states.to(device)
                actions = actions.to(device)
                for k, (policy, opt) in enumerate(zip(policies, optimizers)):
                    loss = F.cross_entropy(policy(states), actions)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                    opt.step()
                    total_losses[k] += loss.item()
                if log_every > 0 and (bidx + 1) % log_every == 0:
                    avg = [f"{l / (bidx+1):.4f}" for l in total_losses]
                    print(f"  [WarmUp] ep {ep+1} step {bidx+1}  avg_loss={avg}")
            avgs = [f"{l / len(loader):.4f}" for l in total_losses]
            print(f"  [WarmUp] Epoch {ep+1}/{warmup_bc_epochs}  avg_loss={avgs}")

    # ── EM loop ───────────────────────────────────────────────────────────────
    eval_history = {"em_iter": [], "style_0": [], "style_1": [], "style_2": []}

    # Initial W: uniform
    W = np.full((dataset.num_trajectories, K), 1.0 / K, dtype=np.float32)

    for em_iter in range(num_em_iters):
        print(f"\n=== EM iteration {em_iter + 1}/{num_em_iters} ===")

        # ── E-step ──────────────────────────────────────────────────────────
        print("  E-step: computing soft assignments …")
        W = compute_soft_assignments(
            policies=policies,
            dataset=dataset,
            device=device,
            temperature=temperature,
        )
        w_ent = float(-(W * np.log(W + 1e-8)).sum(axis=1).mean())
        # Convert to tensors for M-step
        W_t   = torch.from_numpy(W)                       # [N, K] on CPU
        adv_t = torch.from_numpy(advantages)              # [N] on CPU
        print(f"  E-step done. Mean assignment entropy = {w_ent:.4f}")

        # ── M-step ──────────────────────────────────────────────────────────
        print(f"  M-step: {m_step_epochs} epoch(s) …")
        for m_ep in range(m_step_epochs):
            total_losses = [0.0] * K

            for bidx, (states, actions, traj_ids) in enumerate(loader):
                states   = states.to(device)
                actions  = actions.to(device)
                # traj_ids: [B] indices into W_t and adv_t

                adv_b = adv_t[traj_ids].to(device)        # [B]
                adv_w = torch.exp(
                    torch.clamp(adv_b / beta, -adv_clip, adv_clip)
                )
                adv_w = adv_w / adv_w.mean().clamp_min(1e-8)

                for k, (policy, opt) in enumerate(zip(policies, optimizers)):
                    w_k  = W_t[traj_ids, k].to(device)    # [B]
                    wt   = w_k * adv_w                    # [B]

                    logits = policy(states)
                    ce     = F.cross_entropy(
                        logits, actions, reduction="none"
                    )                                      # [B]
                    loss   = (wt * ce).mean()

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                    opt.step()
                    total_losses[k] += loss.item()

                if log_every > 0 and (bidx + 1) % log_every == 0:
                    avg = [f"{l / (bidx+1):.4f}" for l in total_losses]
                    print(
                        f"  [M-step] iter {em_iter+1} ep {m_ep+1}"
                        f" step {bidx+1}  avg_loss={avg}"
                    )

            avgs = [f"{l / len(loader):.4f}" for l in total_losses]
            print(
                f"  [M-step] iter {em_iter+1} ep {m_ep+1}/{m_step_epochs}"
                f"  avg_loss={avgs}"
            )

        # ── Optional online evaluation ──────────────────────────────────────
        if eval_every > 0 and (em_iter + 1) % eval_every == 0:
            print(f"  Online evaluation at EM iter {em_iter + 1} …")
            eval_res = evaluate_sorl(
                policies=policies,
                dataset=dataset,
                num_episodes_per_style=num_eval_ep,
                max_ep_len=max_ep_len,
                device=device,
            )
            eval_history["em_iter"].append(em_iter + 1)
            for k in range(K):
                mean_r = float(np.mean(eval_res[k])) if eval_res[k] else 0.0
                eval_history[f"style_{k}"].append(mean_r)
                print(f"    style_{k}: mean_r={mean_r:.3f}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for k, policy in enumerate(policies):
                sp = os.path.join(save_dir, f"sorl_bc_style{k}.pth")
                torch.save(policy.state_dict(), sp)
            print(f"  Saved {K} policy checkpoints → {save_dir}/sorl_bc_style*.pth")

    if eval_history["em_iter"]:
        _plot_eval_history(eval_history)

    return policies


# =============================================================================
# Online Evaluation
# =============================================================================

def evaluate_sorl(
    policies:               list,
    dataset:                SODataset,
    num_episodes_per_style: int   = 10,
    max_ep_len:             int   = 100,
    device:                 str   = "cpu",
    initial_rtg:            float = 1.0,   # unused; kept for API symmetry
    env_kwargs:             dict  = None,
    style_map:              dict  = None,
) -> dict:
    """
    Roll out each discovered BC policy online.

    style_map: optional {k: env_style_id} — maps discovered SORL style k to
    the MiniGrid target style for evaluation with style-specific bonuses.
    Defaults to identity {k: k}.  After training, inspect W to identify the
    correspondence and pass the correct mapping here.

    Returns {k: [episode_returns]}.
    """
    K = len(policies)
    if env_kwargs is None:
        env_kwargs = {}
    if style_map is None:
        style_map = {k: k for k in range(K)}

    state_mean = torch.tensor(dataset.state_mean, dtype=torch.float32, device=device)
    state_std  = torch.tensor(dataset.state_std,  dtype=torch.float32, device=device)

    results = {k: [] for k in range(K)}

    for k, policy in enumerate(policies):
        policy.eval()
        env_style_id = style_map[k]

        with torch.no_grad():
            for ep in range(num_episodes_per_style):
                env = MiniGridThreeStyles(
                    target_style=STYLE_NAMES[env_style_id],
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
                    ).float().to(device)
                    state = (state - state_mean) / state_std

                    action = policy.get_action(state)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_return += float(reward)
                    t += 1

                results[k].append(episode_return)
                env.close()

        print(
            f"[SORL] Style {k} → env style {env_style_id} ({STYLE_NAMES[env_style_id]}): "
            f"mean={np.mean(results[k]):.3f} ± {np.std(results[k]):.3f}"
        )
        policy.train()

    return results


# =============================================================================
# Plotting
# =============================================================================

def _plot_eval_history(eval_history: dict, save_path: str = "eval_results_sorl.png"):
    K     = len([k for k in eval_history if k.startswith("style_")])
    iters = eval_history["em_iter"]
    plt.figure(figsize=(10, 6))
    for k in range(K):
        plt.plot(
            iters, eval_history[f"style_{k}"],
            marker="o", linewidth=2,
            label=f"SORL Style {k}",
        )
    plt.xlabel("EM Iteration", fontsize=12)
    plt.ylabel("Mean Episode Return", fontsize=12)
    plt.title("SORL — Online Evaluation Returns by Discovered Style", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
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

    SAVE_DIR = "trained_models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Dataset -----------------------------------------------------------
    dataset = SODataset(
        trajectory_paths=paths,
        sampling=True,
        index_channel_only=True,
    )
    STATE_DIM = dataset.state_dim_flat   # 9  (3×3 index channel)
    ACT_DIM   = 7

    # --- Stage 1: Value Network --------------------------------------------
    value_net  = ValueNetwork(state_dim=STATE_DIM, hidden=128)
    value_save = os.path.join(SAVE_DIR, "sorl_value_net.pth")

    if os.path.exists(value_save):
        value_net.load_state_dict(torch.load(value_save, map_location=device))
        print(f"Loaded value network from {value_save}")
    else:
        value_net = train_value_network(
            value_net=value_net,
            dataset=dataset,
            num_epochs=30,
            batch_size=256,
            lr=1e-3,
            device=device,
            save_path=value_save,
        )

    # --- Stage 2: SORL -----------------------------------------------------
    policies = [
        BCPolicy(state_dim=STATE_DIM, act_dim=ACT_DIM, hidden_size=256, num_layers=3)
        for _ in range(NUM_STYLES)
    ]

    sorl_saves = [os.path.join(SAVE_DIR, f"sorl_bc_style{k}.pth") for k in range(NUM_STYLES)]

    if all(os.path.exists(sp) for sp in sorl_saves):
        for k, (policy, sp) in enumerate(zip(policies, sorl_saves)):
            policy.load_state_dict(torch.load(sp, map_location=device))
        print(f"Loaded {NUM_STYLES} SORL policies from {SAVE_DIR}/")
    else:
        policies = train_sorl(
            policies=policies,
            value_net=value_net,
            dataset=dataset,
            num_em_iters=15,
            m_step_epochs=5,
            batch_size=256,
            lr=1e-3,
            grad_clip=1.0,
            device=device,
            beta=1.0,
            adv_clip=5.0,
            temperature=1.0,
            log_every=50,
            save_dir=SAVE_DIR,
            warmup_bc_epochs=5,
            eval_every=3,
            num_eval_ep=10,
            max_ep_len=100,
        )

    # --- Final evaluation --------------------------------------------------
    print("\n=== Final SORL Evaluation ===")
    results = evaluate_sorl(
        policies=policies,
        dataset=dataset,
        num_episodes_per_style=20,
        max_ep_len=100,
        device=device,
    )
    for k in range(NUM_STYLES):
        print(
            f"  Style {k}: mean={np.mean(results[k]):.3f}"
            f" ± {np.std(results[k]):.3f}"
        )