"""
Comprehensive evaluation metrics for StyleVAEPromptDT (controls-conditioned prior).

Metrics
-------
Rollout (online, per style):
  success_rate             % of episodes reaching the goal
  style_achievement_rate   % of successes where achieved_style == target_style
  avg_return               mean episode return
  avg_episode_length       mean steps per episode
  detection_rate           % of episodes terminated by detection
  avg_enemy_distance       proxy for stealth / safety behaviour
  avg_path_efficiency      proxy for commitment / directness
  weapon_usage_rate        % of episodes where weapon was picked up
  camouflage_usage_rate    % of episodes where camouflage was picked up

Control fidelity (requires collecting per-episode control + outcome):
  control_fidelity         mean Spearman r between each control dim and its
                           corresponding behavioural outcome — measures how
                           well the control vector actually drives behaviour

Latent quality (offline, from dataset forward pass):
  encoder_silhouette        sklearn silhouette score of encoder z by style label
  encoder_style_accuracy    logistic-regression accuracy z → style (linear probe)
  prior_silhouette          silhouette score of prior z = mu_p(c) by style label
  mean_kl_divergence        KL( q(z|traj) || p(z|c) ) averaged over dataset

Reconstruction (offline):
  action_accuracy           top-1 action prediction accuracy on the dataset

Usage
-----
Run from repo root:
    python trajectory_embedding/style_dec_vae/transformer/style_pdt_vae/evaluate_metrics.py

Or import and call run_full_evaluation() for programmatic use.
"""

import os
import sys
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths
from trajectory_embedding.style_dec_vae.transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
    StyleVAEPromptDT,
    kl_q_p_diag,
)
from envs.three_style_env import MiniGridThreeStyles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STYLE_NAMES  = {0: "bypass", 1: "weapon", 2: "camouflage"}
STYLE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}

CONTROL_NAMES = [
    "risk_tolerance",
    "resource_pref",
    "stealth_pref",
    "safety_pref",
    "commitment",
]

# Which behavioural outcome is each control dim meant to drive?
# (dim_index, outcome_key, direction) — direction +1 means higher control → higher outcome
CONTROL_OUTCOME_MAP = [
    (0, "inv_min_dist",       +1),   # risk_tolerance   ↔ 1 - norm_min_distance
    (1, "resource_used",      +1),   # resource_pref    ↔ items_picked (norm)
    (2, "stealth_score",      +1),   # stealth_pref     ↔ avg_dist * (1-detected)
    (3, "avg_enemy_distance", +1),   # safety_pref      ↔ avg_enemy_distance (norm)
    (4, "path_efficiency",    +1),   # commitment       ↔ path_efficiency
]
MAX_ENEMY_DIST = 12.0  # normalisation constant (same as controls_from_episode_summary)


# ---------------------------------------------------------------------------
# Per-episode data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    style_id:           int
    target_style:       str
    control_vector:     np.ndarray          # [control_dim]
    episode_return:     float
    length:             int
    success:            bool                # reached goal
    achieved_style:     Optional[str]
    detected:           bool
    # behavioural fields (None when episode ended via detection / timeout)
    avg_enemy_distance: Optional[float] = None
    min_enemy_distance: Optional[float] = None
    path_efficiency:    Optional[float] = None
    items_picked:       Optional[int]   = None
    picked_weapon:      Optional[bool]  = None
    picked_camouflage:  Optional[bool]  = None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout_episode(
    model: StyleVAEPromptDT,
    style_tokens: torch.Tensor,
    env: MiniGridThreeStyles,
    state_mean: torch.Tensor,
    state_std:  torch.Tensor,
    device:     str,
    initial_rtg: float,
    max_ep_len:  int,
    max_context: int,
    seed:        int,
) -> dict:
    """Single episode rollout; returns raw outcome dict."""
    obs, _ = env.reset(seed=seed)

    state = torch.from_numpy(obs["image"][:, :, 0].flatten()).float().to(device)
    state = (state - state_mean) / state_std

    states    = state.reshape(1, 1, -1)
    actions   = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    rtgs      = torch.tensor([[[initial_rtg]]], dtype=torch.float32, device=device)
    timesteps = torch.tensor([[0]], dtype=torch.long, device=device)

    ep_return = 0.0
    done      = False
    t         = 0

    while not done and t < max_ep_len:
        attn_mask = torch.ones((1, states.shape[1]), dtype=torch.float32, device=device)
        _, action_preds, _ = model.dt.forward(
            states=states, actions=actions, returns_to_go=rtgs,
            timesteps=timesteps, attention_mask=attn_mask,
            style_tokens=style_tokens,
        )
        action = torch.argmax(action_preds[:, -1], dim=-1).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += float(reward)
        t += 1

        if not done:
            ns = torch.from_numpy(next_obs["image"][:, :, 0].flatten()).float().to(device)
            ns = (ns - state_mean) / state_std
            states    = torch.cat([states, ns.reshape(1, 1, -1)], dim=1)
            actions   = torch.cat([actions, torch.tensor([[[action]]], dtype=torch.long, device=device)], dim=1)
            rtgs      = torch.cat([rtgs, rtgs[:, -1:] - reward], dim=1)
            timesteps = torch.cat([timesteps, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
            if max_context and states.shape[1] > max_context:
                states    = states[:, -max_context:]
                actions   = actions[:, -max_context:]
                rtgs      = rtgs[:, -max_context:]
                timesteps = timesteps[:, -max_context:]

    # Parse info
    es = info.get("episode_summary")
    detected = info.get("detected", False) or info.get("termination") == "detected_front_tile"
    success  = (es is not None)

    return dict(
        episode_return=ep_return,
        length=t,
        success=success,
        achieved_style=es.get("achieved_style") if es else None,
        detected=detected,
        avg_enemy_distance=es.get("avg_enemy_distance") if es else None,
        min_enemy_distance=es.get("min_enemy_distance") if es else None,
        path_efficiency=es.get("path_efficiency") if es else None,
        items_picked=es.get("items_picked") if es else None,
        picked_weapon=es.get("picked_weapon") if es else None,
        picked_camouflage=es.get("picked_camouflage") if es else None,
    )


def run_rollout_evaluation(
    model:                  StyleVAEPromptDT,
    dataset:                MiniGridDataset,
    num_episodes_per_style: int  = 30,
    device:                 str  = "cpu",
    initial_rtg:            float = 1.0,
    max_ep_len:             int  = 100,
    max_context:            int  = 20,
    env_kwargs:             dict = None,
    deterministic_prior:    bool = False,
    num_control_samples:    int  = 5,
) -> List[EpisodeRecord]:
    """
    For each style, sample `num_control_samples` control vectors from the dataset,
    generate style tokens via the prior, and run `num_episodes_per_style` rollouts
    per control vector.  Returns a flat list of EpisodeRecord objects.
    """
    if env_kwargs is None:
        env_kwargs = {}

    model.eval()
    state_mean = torch.tensor(dataset.state_mean, device=device, dtype=torch.float32)
    state_std  = torch.tensor(dataset.state_std,  device=device, dtype=torch.float32)

    records: List[EpisodeRecord] = []
    seed_counter = 0

    with torch.no_grad():
        for style_id, style_name in STYLE_NAMES.items():

            # --- pick representative control vectors from dataset for this style ---
            style_idx = [i for i, t in enumerate(dataset.tasks) if t == style_id]
            if len(style_idx) == 0:
                print(f"  Warning: no dataset trajectories for style {style_name}")
                continue
            sampled_idx = random.sample(style_idx, min(num_control_samples, len(style_idx)))
            control_vectors = [dataset.controls[i] for i in sampled_idx]

            for c_np in control_vectors:
                c_tensor = torch.tensor(c_np, dtype=torch.float32, device=device).unsqueeze(0)
                z = model.sample_z_from_prior(c_tensor, deterministic=deterministic_prior)
                style_tokens = model.latent_to_style_tokens(z)

                env = MiniGridThreeStyles(
                    target_style=style_name,
                    target_bonus=1.0,
                    non_target_penalty=-1.0,
                    easy_env=False,
                    agent_view_size=3,
                    **env_kwargs,
                )

                for ep in range(num_episodes_per_style):
                    outcome = _rollout_episode(
                        model, style_tokens, env,
                        state_mean, state_std, device,
                        initial_rtg, max_ep_len, max_context,
                        seed=seed_counter,
                    )
                    seed_counter += 1
                    records.append(EpisodeRecord(
                        style_id=style_id,
                        target_style=style_name,
                        control_vector=c_np.copy(),
                        **outcome,
                    ))
                env.close()

    model.train()
    return records


# ---------------------------------------------------------------------------
# Aggregate rollout metrics
# ---------------------------------------------------------------------------

def aggregate_rollout_metrics(records: List[EpisodeRecord]) -> dict:
    """
    Aggregate EpisodeRecord list into per-style and overall metric dicts.
    Returns:
        {
          "per_style": {style_name: {metric: value, ...}, ...},
          "overall":   {metric: value, ...},
        }
    """
    def _agg(recs):
        n = len(recs)
        if n == 0:
            return {}
        successes  = [r for r in recs if r.success]
        n_suc      = len(successes)
        achieved   = [r for r in successes if r.achieved_style == r.target_style]

        returns       = [r.episode_return for r in recs]
        lengths       = [r.length for r in recs]
        detected      = [r.detected for r in recs]

        def _mean_field(recs_list, field_name):
            vals = [getattr(r, field_name) for r in recs_list if getattr(r, field_name) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "n_episodes":             n,
            "success_rate":           n_suc / n,
            "style_achievement_rate": len(achieved) / n_suc if n_suc > 0 else float("nan"),
            "avg_return":             float(np.mean(returns)),
            "avg_episode_length":     float(np.mean(lengths)),
            "detection_rate":         float(np.mean(detected)),
            "avg_enemy_distance":     _mean_field(successes, "avg_enemy_distance"),
            "avg_path_efficiency":    _mean_field(successes, "path_efficiency"),
            "weapon_usage_rate":      float(np.mean([r.picked_weapon  for r in successes])) if successes else float("nan"),
            "camouflage_usage_rate":  float(np.mean([r.picked_camouflage for r in successes])) if successes else float("nan"),
        }

    per_style = {}
    for sid, sname in STYLE_NAMES.items():
        per_style[sname] = _agg([r for r in records if r.style_id == sid])

    return {
        "per_style": per_style,
        "overall":   _agg(records),
    }


# ---------------------------------------------------------------------------
# Control fidelity
# ---------------------------------------------------------------------------

def compute_control_fidelity(records: List[EpisodeRecord]) -> dict:
    """
    Compute Spearman correlation between each control dimension and its intended
    behavioural outcome across all successful episodes.

    Returns dict with per-dimension r, p-value, and mean_r.
    """
    suc = [r for r in records if r.success
           and r.avg_enemy_distance is not None
           and r.path_efficiency is not None
           and r.items_picked is not None]

    if len(suc) < 5:
        print("  Warning: too few successful episodes for control fidelity.")
        return {"mean_spearman_r": float("nan"), "per_dim": {}}

    # Build outcome arrays (normalised to [0,1])
    outcomes = {
        "inv_min_dist":       np.array([1.0 - min(r.min_enemy_distance / MAX_ENEMY_DIST, 1.0) for r in suc]),
        "resource_used":      np.clip(np.array([r.items_picked / 2.0 for r in suc]), 0, 1),
        "stealth_score":      np.clip(
                                  np.array([r.avg_enemy_distance / MAX_ENEMY_DIST for r in suc])
                                  * np.array([1.0 - float(r.detected) for r in suc]),
                              0, 1),
        "avg_enemy_distance": np.clip(np.array([r.avg_enemy_distance / MAX_ENEMY_DIST for r in suc]), 0, 1),
        "path_efficiency":    np.array([r.path_efficiency for r in suc]),
    }

    results = {}
    spearman_rs = []
    for dim_idx, outcome_key, _ in CONTROL_OUTCOME_MAP:
        ctrl_vals = np.array([r.control_vector[dim_idx] for r in suc])
        out_vals  = outcomes[outcome_key]
        r_val, p_val = spearmanr(ctrl_vals, out_vals)
        dim_name = CONTROL_NAMES[dim_idx]
        results[dim_name] = {"spearman_r": float(r_val), "p_value": float(p_val)}
        spearman_rs.append(abs(float(r_val)))

    results["mean_abs_spearman_r"] = float(np.mean(spearman_rs))
    return results


# ---------------------------------------------------------------------------
# Latent quality (offline)
# ---------------------------------------------------------------------------

def compute_latent_metrics(
    model:   StyleVAEPromptDT,
    dataset: MiniGridDataset,
    device:  str = "cpu",
    batch_size: int = 64,
) -> dict:
    """
    Offline metrics on the full dataset:
      - encoder silhouette score
      - linear probe accuracy on encoder z
      - prior silhouette score (z sampled from p(z|c))
      - mean KL divergence
      - action reconstruction accuracy
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    all_z_enc   = []
    all_z_prior = []
    all_labels  = []
    all_kl      = []
    n_correct   = 0
    n_total     = 0

    with torch.no_grad():
        for batch in loader:
            full_states   = batch["full_states"].to(device)
            full_actions  = batch["full_actions"].to(device)
            full_timesteps= batch["full_timesteps"].to(device)
            full_mask     = batch["full_attention_mask"].to(device)
            controls      = batch["controls"].to(device)
            states        = batch["states"].to(device)
            actions       = batch["actions"].to(device)
            rtgs          = batch["returns_to_go"].to(device)
            timesteps     = batch["timesteps"].to(device)
            attn_mask     = batch["attention_mask"].to(device)
            labels        = batch["task_labels"]

            # --- encoder posterior ---
            mu_q, logvar_q, z_enc = model.encode_full_trajectory(
                full_states, full_actions, full_timesteps, full_mask
            )

            # --- prior ---
            mu_p, logvar_p = model.prior(controls)
            z_prior = mu_p  # deterministic (mean of prior)

            # --- KL ---
            kl = kl_q_p_diag(mu_q, logvar_q, mu_p, logvar_p)
            all_kl.extend(kl.cpu().numpy().tolist())

            # --- action accuracy using encoder z ---
            style_tokens = model.latent_to_style_tokens(z_enc)
            _, action_preds, _ = model.dt(
                states=states, actions=actions, returns_to_go=rtgs,
                timesteps=timesteps, attention_mask=attn_mask,
                style_tokens=style_tokens,
            )
            B, T, C = action_preds.shape
            acts_ce = actions.squeeze(-1).long()
            acts_ce = torch.clamp(acts_ce, 0, C - 1)
            predicted = torch.argmax(action_preds, dim=-1)  # [B, T]
            mask_bool = attn_mask.bool()
            n_correct += (predicted[mask_bool] == acts_ce[mask_bool]).sum().item()
            n_total   += mask_bool.sum().item()

            all_z_enc.extend(z_enc.cpu().numpy())
            all_z_prior.extend(z_prior.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    Z_enc   = np.array(all_z_enc)
    Z_prior = np.array(all_z_prior)
    Y       = np.array(all_labels)

    results = {}

    # --- silhouette scores ---
    results["encoder_silhouette"] = float(silhouette_score(Z_enc, Y))
    results["prior_silhouette"]   = float(silhouette_score(Z_prior, Y))

    # --- linear probe on encoder z ---
    scaler  = StandardScaler()
    Z_scaled = scaler.fit_transform(Z_enc)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    # 80/20 split
    n = len(Y)
    split = int(0.8 * n)
    idx = np.random.permutation(n)
    clf.fit(Z_scaled[idx[:split]], Y[idx[:split]])
    acc = clf.score(Z_scaled[idx[split:]], Y[idx[split:]])
    results["encoder_style_accuracy"] = float(acc)

    # --- KL and action accuracy ---
    results["mean_kl_divergence"] = float(np.mean(all_kl))
    results["action_accuracy"]    = float(n_correct / n_total) if n_total > 0 else float("nan")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model:                  StyleVAEPromptDT,
    dataset:                MiniGridDataset,
    device:                 str  = "cpu",
    num_episodes_per_style: int  = 30,
    num_control_samples:    int  = 5,
    max_ep_len:             int  = 100,
    initial_rtg:            float = 1.0,
    max_context:            int  = 20,
    env_kwargs:             dict = None,
    deterministic_prior:    bool = False,
) -> dict:
    """
    Run all metrics and return a single unified dict.
    """
    print("=== Rollout evaluation ===")
    records = run_rollout_evaluation(
        model, dataset,
        num_episodes_per_style=num_episodes_per_style,
        num_control_samples=num_control_samples,
        device=device,
        initial_rtg=initial_rtg,
        max_ep_len=max_ep_len,
        max_context=max_context,
        env_kwargs=env_kwargs,
        deterministic_prior=deterministic_prior,
    )

    rollout = aggregate_rollout_metrics(records)

    print("=== Control fidelity ===")
    fidelity = compute_control_fidelity(records)

    print("=== Latent / reconstruction metrics ===")
    latent = compute_latent_metrics(model, dataset, device=device)

    return {
        "rollout":          rollout,
        "control_fidelity": fidelity,
        "latent":           latent,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, model_name: str = "StyleVAEPromptDT"):
    """Print a readable summary of all metrics."""
    bar = "=" * 60

    print(f"\n{bar}")
    print(f"  {model_name}")
    print(bar)

    # --- rollout ---
    print("\n[ Rollout metrics ]")
    header = f"{'metric':<30} {'overall':>10}" + "".join(f"  {n:>12}" for n in STYLE_NAMES.values())
    print(header)
    print("-" * len(header))
    rollout_keys = [
        "success_rate", "style_achievement_rate", "avg_return",
        "avg_episode_length", "detection_rate",
        "avg_enemy_distance", "avg_path_efficiency",
        "weapon_usage_rate", "camouflage_usage_rate",
    ]
    for k in rollout_keys:
        ov = metrics["rollout"]["overall"].get(k, float("nan"))
        row = f"{k:<30} {ov:>10.3f}"
        for sname in STYLE_NAMES.values():
            val = metrics["rollout"]["per_style"].get(sname, {}).get(k, float("nan"))
            row += f"  {val:>12.3f}"
        print(row)

    # --- control fidelity ---
    print("\n[ Control fidelity  (Spearman r, higher = controls drive behaviour) ]")
    cf = metrics["control_fidelity"]
    for dim_name in CONTROL_NAMES:
        info = cf.get(dim_name, {})
        r  = info.get("spearman_r", float("nan"))
        p  = info.get("p_value",   float("nan"))
        sig = "*" if p < 0.05 else " "
        print(f"  {dim_name:<22}  r = {r:+.3f}  (p={p:.3f}){sig}")
    print(f"  {'mean |r|':<22}  {cf.get('mean_abs_spearman_r', float('nan')):.3f}")

    # --- latent ---
    print("\n[ Latent / reconstruction metrics ]")
    lt = metrics["latent"]
    for k, v in lt.items():
        print(f"  {k:<35}  {v:.4f}")

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Comparison table (multiple models)
# ---------------------------------------------------------------------------

def print_comparison_table(all_metrics: dict):
    """
    all_metrics: {model_name: metrics_dict}
    Prints a side-by-side table of key scalar metrics.
    """
    key_metrics = [
        ("overall / success_rate",           lambda m: m["rollout"]["overall"].get("success_rate", float("nan"))),
        ("overall / style_achievement_rate", lambda m: m["rollout"]["overall"].get("style_achievement_rate", float("nan"))),
        ("overall / avg_return",             lambda m: m["rollout"]["overall"].get("avg_return", float("nan"))),
        ("overall / detection_rate",         lambda m: m["rollout"]["overall"].get("detection_rate", float("nan"))),
        ("control_fidelity / mean_abs_r",    lambda m: m["control_fidelity"].get("mean_abs_spearman_r", float("nan"))),
        ("latent / encoder_silhouette",      lambda m: m["latent"].get("encoder_silhouette", float("nan"))),
        ("latent / encoder_style_accuracy",  lambda m: m["latent"].get("encoder_style_accuracy", float("nan"))),
        ("latent / mean_kl_divergence",      lambda m: m["latent"].get("mean_kl_divergence", float("nan"))),
        ("latent / action_accuracy",         lambda m: m["latent"].get("action_accuracy", float("nan"))),
    ]

    model_names = list(all_metrics.keys())
    col_w = max(20, max(len(n) for n in model_names))
    print("\n" + "=" * (34 + col_w * len(model_names)))
    print(f"{'metric':<34}" + "".join(f"{n:>{col_w}}" for n in model_names))
    print("-" * (34 + col_w * len(model_names)))
    for label, extractor in key_metrics:
        row = f"{label:<34}"
        for mname in model_names:
            val = extractor(all_metrics[mname])
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * (34 + col_w * len(model_names)) + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_evaluation(metrics: dict, model_name: str = "StyleVAEPromptDT", save_dir: str = None):
    """
    Produces two figures:
      eval_rollout.png   – per-style bar charts of key rollout metrics
      eval_latent.png    – control fidelity + latent/reconstruction metrics
    """
    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    style_names = list(STYLE_NAMES.values())
    colors = [STYLE_COLORS[s] for s in STYLE_NAMES]

    # ---- Figure 1: rollout metrics ----------------------------------------
    rollout_show = [
        ("success_rate",           "Success rate"),
        ("style_achievement_rate", "Style achievement rate"),
        ("avg_return",             "Avg episode return"),
        ("detection_rate",         "Detection rate"),
        ("avg_enemy_distance",     "Avg enemy distance"),
        ("avg_path_efficiency",    "Avg path efficiency"),
    ]
    n_plots = len(rollout_show)
    fig1, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig1.suptitle(f"{model_name} — Rollout Metrics per Style", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, rollout_show):
        vals = [metrics["rollout"]["per_style"].get(sn, {}).get(key, float("nan")) for sn in style_names]
        bars = ax.bar(style_names, vals, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, max(1.1, max(v for v in vals if not np.isnan(v)) * 1.2) if any(not np.isnan(v) for v in vals) else 1.1)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p = os.path.join(save_dir, "plots/eval_rollout.png")
    plt.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close()

    # ---- Figure 2: control fidelity + latent metrics ----------------------
    fig2 = plt.figure(figsize=(14, 5))
    fig2.suptitle(f"{model_name} — Control Fidelity & Latent Quality", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

    # Left: control fidelity spider / bar
    ax_cf = fig2.add_subplot(gs[0])
    cf = metrics["control_fidelity"]
    r_vals  = [cf.get(d, {}).get("spearman_r", float("nan")) for d in CONTROL_NAMES]
    p_vals  = [cf.get(d, {}).get("p_value",    1.0)          for d in CONTROL_NAMES]
    bar_colors = ["#2ca02c" if (not np.isnan(r) and p < 0.05) else "#d62728" if not np.isnan(r) else "#aaaaaa"
                  for r, p in zip(r_vals, p_vals)]
    x = np.arange(len(CONTROL_NAMES))
    bars = ax_cf.bar(x, r_vals, color=bar_colors, alpha=0.85)
    ax_cf.axhline(0, color="black", linewidth=0.8)
    ax_cf.set_xticks(x)
    ax_cf.set_xticklabels([n.replace("_", "\n") for n in CONTROL_NAMES], fontsize=8)
    ax_cf.set_ylim(-1.05, 1.05)
    ax_cf.set_ylabel("Spearman r", fontsize=9)
    ax_cf.set_title(f"Control Fidelity\n(mean |r| = {cf.get('mean_abs_spearman_r', float('nan')):.3f})", fontsize=10)
    ax_cf.yaxis.grid(True, alpha=0.4)
    ax_cf.set_axisbelow(True)
    # legend
    from matplotlib.patches import Patch
    ax_cf.legend(handles=[Patch(color="#2ca02c", label="significant (p<0.05)"),
                           Patch(color="#d62728", label="non-significant")],
                 fontsize=8, loc="lower right")

    # Right: latent + reconstruction metrics
    ax_lt = fig2.add_subplot(gs[1])
    lt = metrics["latent"]
    lt_keys   = ["encoder_silhouette", "prior_silhouette", "encoder_style_accuracy", "action_accuracy"]
    lt_labels = ["Enc. silhouette", "Prior silhouette", "Style accuracy\n(linear probe)", "Action accuracy"]
    lt_vals   = [lt.get(k, float("nan")) for k in lt_keys]
    lt_colors = ["#4C72B0", "#9467BD", "#55A868", "#DD8452"]
    bars2 = ax_lt.bar(lt_labels, lt_vals, color=lt_colors, alpha=0.85)
    ax_lt.set_ylim(0, 1.15)
    ax_lt.set_title("Latent & Reconstruction Quality", fontsize=10)
    ax_lt.yaxis.grid(True, alpha=0.4)
    ax_lt.set_axisbelow(True)
    ax_lt.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars2, lt_vals):
        if not np.isnan(val):
            ax_lt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    # KL separately on twin axis
    ax_kl = ax_lt.twinx()
    kl_val = lt.get("mean_kl_divergence", float("nan"))
    ax_kl.set_ylabel("KL divergence", color="#8c564b", fontsize=8)
    ax_kl.tick_params(axis="y", labelcolor="#8c564b")
    if not np.isnan(kl_val):
        ax_kl.axhline(kl_val, color="#8c564b", linestyle="--", linewidth=1.5,
                      label=f"Mean KL = {kl_val:.4f}")
        ax_kl.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p2 = os.path.join(save_dir, "plots/eval_latent.png")
    plt.savefig(p2, dpi=150)
    print(f"Saved {p2}")
    plt.close()


def plot_comparison(all_metrics: dict, save_dir: str = None):
    """
    all_metrics: {model_name: metrics_dict}
    Grouped bar chart of key metrics across models.
    """
    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    model_names = list(all_metrics.keys())
    n_models = len(model_names)

    compare_metrics = [
        ("overall / success_rate",          "Success rate",          lambda m: m["rollout"]["overall"].get("success_rate", float("nan"))),
        ("overall / style_achievement_rate","Style achievement rate", lambda m: m["rollout"]["overall"].get("style_achievement_rate", float("nan"))),
        ("overall / avg_return",            "Avg return",            lambda m: m["rollout"]["overall"].get("avg_return", float("nan"))),
        ("control_fidelity / mean_abs_r",   "Control fidelity",      lambda m: m["control_fidelity"].get("mean_abs_spearman_r", float("nan"))),
        ("latent / encoder_silhouette",     "Encoder silhouette",    lambda m: m["latent"].get("encoder_silhouette", float("nan"))),
        ("latent / style_accuracy",         "Style accuracy",        lambda m: m["latent"].get("encoder_style_accuracy", float("nan"))),
        ("latent / action_accuracy",        "Action accuracy",       lambda m: m["latent"].get("action_accuracy", float("nan"))),
    ]

    n_metrics = len(compare_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    palette = plt.cm.tab10.colors

    for ax, (_, label, extractor) in zip(axes, compare_metrics):
        vals = [extractor(all_metrics[mn]) for mn in model_names]
        bars = ax.bar(model_names, vals,
                      color=[palette[i % 10] for i in range(n_models)],
                      alpha=0.85)
        ax.set_title(label, fontsize=9)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    p = os.path.join(save_dir, "eval_comparison.png")
    plt.savefig(p, dpi=150)
    print(f"Saved {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEVICE = "cpu"

    dataset_params = {
        "sampling": False,
        "index_channel_only": True,
        "state_normalization_factor": 1,
        "action_normalization_factor": 1,
        "max_len": 20,
        "control_dim": 5,
    }

    print("Loading dataset …")
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)

    model = StyleVAEPromptDT(
        state_dim=9,
        act_dim=7,
        hidden_size=128,
        latent_dim=16,
        max_length=20,
        max_ep_len=100,
        action_tanh=False,
        beta=0.0085,
        control_dim=5,
        prior_hidden=128,
        free_bits=0.0,
        n_layer=4,
        n_head=8,
    )

    CHECKPOINT = os.path.join(
        os.path.dirname(__file__),
        "trained_models/style_prompt_dt_minigrid_controls_condprior.pth",
    )
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Loaded checkpoint: {CHECKPOINT}")
    else:
        print(f"No checkpoint found at {CHECKPOINT} — running with random weights.")

    model.to(DEVICE)

    metrics = run_full_evaluation(
        model, dataset,
        device=DEVICE,
        num_episodes_per_style=30,
        num_control_samples=5,
        max_ep_len=100,
        initial_rtg=1.0,
        max_context=20,
        deterministic_prior=False,
    )

    print_metrics_table(metrics)

    # Save JSON
    save_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"Saved metrics to {save_path}")

    plot_evaluation(metrics)