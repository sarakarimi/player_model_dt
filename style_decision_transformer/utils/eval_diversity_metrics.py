"""
Quantitative + qualitative trajectory-diversity evaluation for StyleDT.

For a FIXED control vector c we sample N trajectories and measure how diverse
they are while staying style-consistent:

  metric 1  D_traj      mean pairwise DTW distance between the N trajectories
                        (grid cells / aligned step)  -- higher = more diverse
  metric 2  H_route     Shannon entropy (bits) of the discrete route mode
                        {upper, middle, lower} taken across the N samples
  paired    SAR         style-achievement rate over the same N samples
                        (diversity only counts if consistency is preserved)

StyleDT draws a fresh z ~ p(z|c) per sample (structural diversity); the
baselines are deterministic given c, so they get their fair shot via stochastic
action sampling at a temperature. Everything is averaged over several layout
seeds for error bars.

Outputs (under utils/diversity_metrics/):
  - diversity_results.json        all numbers
  - traj_diversity_<style>.png    qualitative overlay (one subplot per model)
  - diversity_vs_consistency.png  D_traj (x) vs SAR (y) scatter

Run:
    python style_decision_transformer/utils/eval_diversity_metrics.py
    python style_decision_transformer/utils/eval_diversity_metrics.py --n_z 20 --n_seeds 5
"""

import argparse
import json
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from envs.multi_style_env import MiniGridMultiStyles
from envs.four_style_env import MiniGridFourStyles
from style_decision_transformer.style_pdt_vae.paths import paths
from style_decision_transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
    StyleVAEPromptDT,
)
from style_decision_transformer.style_pdt_vae.control_prompt_pdt import (
    ControlConditionedDT,
    CONTROL_DIM,
)

HERE = os.path.dirname(__file__)
DEVICE = "cpu"

# Four-style ordering (must match paths.py / dataset task ids).
STYLE_ORDER = {0: "portal", 1: "weapon", 2: "camouflage", 3: "daredevil"}

# Keys are correct at import time (so --styles default works); the placeholder
# vectors are overwritten with per-style dataset means inside main() once the
# dataset is loaded.
CANONICAL = {
    "portal":     np.array([0.35, 0.19, 0.79], dtype=np.float32),
    "weapon":     np.array([1.00, 0.06, 0.55], dtype=np.float32),
    "camouflage": np.array([0.37, 0.42, 0.74], dtype=np.float32),
    "daredevil":  np.array([0.58, 0.47, 0.64], dtype=np.float32),
}

GRID_SIZE   = 13                      # MiniGridFourStyles.size
CENTER_ROW  = (GRID_SIZE - 1) / 2.0   # 6.0
INITIAL_RTG = 2.5
MAX_EP_LEN  = 100
DTW_MAX_PTS = 64                      # subsample long paths before DTW


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _subsample(path, max_pts=DTW_MAX_PTS):
    if len(path) <= max_pts:
        return np.asarray(path, dtype=np.float64)
    idx = np.linspace(0, len(path) - 1, max_pts).round().astype(int)
    return np.asarray(path, dtype=np.float64)[idx]


def dtw_distance(a, b):
    """Normalised DTW between two (x, y) position sequences (cells / step)."""
    a, b = _subsample(a), _subsample(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        for j in range(1, m + 1):
            D[i, j] = ci[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def mean_pairwise_dtw(paths):
    paths = [p for p in paths if len(p) >= 2]
    if len(paths) < 2:
        return 0.0
    ds = [dtw_distance(paths[i], paths[j])
          for i in range(len(paths)) for j in range(i + 1, len(paths))]
    return float(np.mean(ds))


def route_label(path):
    """Discrete route mode from the half of the map the path occupies."""
    if not path:
        return "middle"
    ys = np.array([y for _, y in path], dtype=np.float64)
    up = float(np.mean(ys < CENTER_ROW))
    lo = float(np.mean(ys > CENTER_ROW))
    if max(up, lo) < 0.12:
        return "middle"
    return "upper" if up >= lo else "lower"


def route_entropy(paths):
    labels = [route_label(p) for p in paths]
    if not labels:
        return 0.0, {}
    counts = {k: labels.count(k) for k in set(labels)}
    n = len(labels)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return float(h), counts


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #

def make_env(seed):
    env = MiniGridFourStyles(
        target_style=None, target_bonus=1.0, non_target_penalty=-1.0,
        agent_view_size=3, randomize_layout=True, eval_mode=True,
        max_steps=MAX_EP_LEN, render_mode="rgb_array",
    )
    env.reset(seed=seed)
    return env


def rollout(get_action_fn, state_mean, state_std, seed, max_context):
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    map_img = env.render()

    sm = torch.tensor(state_mean, dtype=torch.float32)
    ss = torch.tensor(state_std, dtype=torch.float32)

    def enc(o):
        s = torch.from_numpy(o["image"][:, :, 0].flatten()).float()
        return (s - sm) / ss

    states    = enc(obs).reshape(1, 1, -1)
    actions   = torch.zeros((1, 1, 1), dtype=torch.long)
    rtgs      = torch.tensor([[[INITIAL_RTG]]], dtype=torch.float32)
    timesteps = torch.tensor([[0]], dtype=torch.long)

    positions = [tuple(env.agent_pos)]
    last_info, done, t = {}, False, 0
    while not done and t < MAX_EP_LEN:
        attn = torch.ones((1, states.shape[1]), dtype=torch.float32)
        a = get_action_fn(states, actions, rtgs, timesteps, attn)
        obs, reward, term, trunc, last_info = env.step(a)
        done = term or trunc
        t += 1
        positions.append(tuple(env.agent_pos))
        if not done:
            states    = torch.cat([states, enc(obs).reshape(1, 1, -1)], dim=1)
            actions   = torch.cat([actions, torch.tensor([[[a]]], dtype=torch.long)], dim=1)
            rtgs      = torch.cat([rtgs, rtgs[:, -1:] - reward], dim=1)
            timesteps = torch.cat([timesteps, torch.tensor([[t]], dtype=torch.long)], dim=1)
            if states.shape[1] > max_context:
                states    = states[:, -max_context:]
                actions   = actions[:, -max_context:]
                rtgs      = rtgs[:, -max_context:]
                timesteps = timesteps[:, -max_context:]

    env.close()
    achieved = (last_info.get("achieved_style")
                or last_info.get("episode_summary", {}).get("achieved_style"))
    return positions, achieved, map_img


def _sample_action(logits, temperature):
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    p = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(p, 1).item())


def styledt_action_fn(vae_model, ctrl_t):
    z = vae_model.sample_z_from_prior(ctrl_t, deterministic=False)
    tok = vae_model.latent_to_style_tokens(z)

    def fn(states, actions, rtgs, timesteps, attn):
        with torch.no_grad():
            _, ap, _ = vae_model.dt(
                states=states, actions=actions.squeeze(-1), returns_to_go=rtgs,
                timesteps=timesteps, attention_mask=attn, style_tokens=tok,
            )
        return int(torch.argmax(ap[:, -1], dim=-1).item())
    return fn


def controldt_action_fn(ctrl_model, ctrl_t, temperature):
    def fn(states, actions, rtgs, timesteps, attn):
        with torch.no_grad():
            _, ap, _ = ctrl_model.forward(
                states=states, actions=actions.squeeze(-1), rewards=None,
                returns_to_go=rtgs, timesteps=timesteps, controls=ctrl_t,
                attention_mask=attn,
            )
        return _sample_action(ap[:, -1], temperature)
    return fn


# --------------------------------------------------------------------------- #
# collection + aggregation
# --------------------------------------------------------------------------- #

def collect_group(model_name, models, ctrl_vec, target_style,
                  state_mean, state_std, seed, n_z, temperature, max_context):
    """N rollouts for one (model, style, layout-seed). Returns paths, achieved, map."""
    vae_model, ctrl_model = models
    ctrl_t = torch.tensor(ctrl_vec, dtype=torch.float32).unsqueeze(0)
    paths, achieved, map_img = [], [], None
    for _ in range(n_z):
        if model_name == "StyleDT":
            fn = styledt_action_fn(vae_model, ctrl_t)
        else:
            fn = controldt_action_fn(ctrl_model, ctrl_t, temperature)
        pos, ach, img = rollout(fn, state_mean, state_std, seed, max_context)
        paths.append(pos)
        achieved.append(ach)
        if map_img is None:
            map_img = img
    return paths, achieved, map_img


def group_metrics(paths, achieved, target_style):
    # Diversity (D_traj, H_route) is measured only over style-consistent
    # trajectories: those that achieved the style they were prompted for.
    consistent = [p for p, a in zip(paths, achieved) if a == target_style]
    h_route, route_counts = route_entropy(consistent)
    d_traj = mean_pairwise_dtw(consistent)
    sar = float(np.mean([a == target_style for a in achieved])) if achieved else 0.0
    return {
        "D_traj": d_traj,          # raw diversity AMONG on-style trajectories
        "H_route": h_route,        # raw route entropy among on-style trajectories
        "SAR": sar,
        # SAR-weighted ("usable") diversity: a model is credited for diversity
        # only to the extent it actually stays on-style. Low SAR shrinks it, so a
        # model that is diverse only because it wanders off-style is penalised.
        "D_traj_eff": sar * d_traj,
        "H_route_eff": sar * h_route,
        "n_consistent": len(consistent),
        "route_counts": route_counts,
    }


def aggregate(per_seed):
    out = {}
    for key in ("D_traj", "H_route", "SAR", "D_traj_eff", "H_route_eff"):
        vals = np.array([s[key] for s in per_seed], dtype=np.float64)
        out[key + "_mean"] = float(vals.mean())
        out[key + "_std"] = float(vals.std())
    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #

def plot_overlay(style, model_to_paths_map, save_dir, n_z):
    models = list(model_to_paths_map.keys())
    cmap = plt.cm.plasma
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)
    axes = axes[0]
    fig.suptitle(
        f"Trajectory diversity — {style.capitalize()}  "
        f"({n_z} samples of $z \\sim p_\\psi(z\\mid c)$, fixed layout)",
        fontsize=12, fontweight="bold",
    )
    for ax, name in zip(axes, models):
        paths, map_img = model_to_paths_map[name]
        if map_img is not None:
            ax.imshow(map_img, origin="upper")
            h, w = map_img.shape[:2]
        else:
            h = w = GRID_SIZE * 32
            ax.set_facecolor("#e8e8e8")
        cell_w, cell_h = w / GRID_SIZE, h / GRID_SIZE   # full grid incl. border walls
        n = len(paths)
        for i, traj in enumerate(paths):
            if len(traj) < 2:
                continue
            color = cmap(i / max(n - 1, 1))
            xs = [(x + 0.5) * cell_w for x, _ in traj]
            ys = [(y + 0.5) * cell_h for _, y in traj]
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.75)
            ax.plot(xs[0], ys[0], "o", color=color, markersize=5, zorder=5)
            ax.plot(xs[-1], ys[-1], "*", color=color, markersize=9, zorder=5)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(save_dir, f"traj_diversity_{style}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def plot_scatter(table, save_dir):
    markers = {"StyleDT": "*", "ControlDT": "o", "PromptDT": "s"}
    styles = list(table.keys())
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i) for i, s in enumerate(styles)}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for style in styles:
        for model, agg in table[style].items():
            ax.errorbar(
                agg["D_traj_mean"], agg["SAR_mean"],
                xerr=agg["D_traj_std"], yerr=agg["SAR_std"],
                marker=markers.get(model, "x"), color=color[style],
                markersize=12 if model == "StyleDT" else 9,
                capsize=3, linestyle="none",
                label=f"{model} · {style}",
            )
    ax.set_xlabel("Trajectory diversity  $D_{traj}$  (DTW cells / step)  →")
    ax.set_ylabel("Style-achievement rate (SAR)  →")
    ax.set_title("Diversity vs. consistency\n(top-right = diverse AND on-style)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    plt.tight_layout()
    out = os.path.join(save_dir, "diversity_vs_consistency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _infer_max_ep_len(state_dict, default=100):
    for k, v in state_dict.items():
        if k.endswith("embed_timestep.weight"):
            return int(v.shape[0])
    return default


def load_models():
    vae_ckpt = os.path.join(HERE, "..", "style_pdt_vae", "trained_models",
                            "style_prompt_dt_minigrid_controls_condprior.pth")
    vae_mel = 100
    if os.path.exists(vae_ckpt):
        vae_sd = torch.load(vae_ckpt, map_location=DEVICE)
        vae_mel = _infer_max_ep_len(vae_sd)
    vae = StyleVAEPromptDT(
        state_dim=9, act_dim=7, hidden_size=128, latent_dim=16, max_length=20,
        max_ep_len=vae_mel, action_tanh=False, beta=0.0085, control_dim=CONTROL_DIM,
        prior_hidden=128, free_bits=0.0, n_layer=4, n_head=8,
    )
    if os.path.exists(vae_ckpt):
        vae.load_state_dict(vae_sd)
        print(f"Loaded StyleDT (max_ep_len={vae_mel}): {vae_ckpt}")
    else:
        print(f"WARNING: StyleDT checkpoint missing ({vae_ckpt}) — random weights.")
    vae.to(DEVICE).eval()

    ctrl_ckpt = os.path.join(HERE, "..", "style_pdt_vae", "trained_models",
                             "control_dt_minigrid.pth")
    ctrl_mel = 100
    if os.path.exists(ctrl_ckpt):
        ctrl_sd = torch.load(ctrl_ckpt, map_location=DEVICE)
        ctrl_mel = _infer_max_ep_len(ctrl_sd)
    ctrl = ControlConditionedDT(
        state_dim=9, act_dim=7, hidden_size=128, control_dim=CONTROL_DIM,
        max_length=8, max_ep_len=ctrl_mel, action_tanh=False, n_layer=4, n_head=8,
    )
    if os.path.exists(ctrl_ckpt):
        ctrl.load_state_dict(ctrl_sd)
        print(f"Loaded ControlDT (max_ep_len={ctrl_mel}): {ctrl_ckpt}")
    else:
        print(f"WARNING: ControlDT checkpoint missing ({ctrl_ckpt}) — random weights.")
    ctrl.to(DEVICE).eval()

    ctx = {"StyleDT": getattr(vae.dt, "max_length", 20),
           "ControlDT": getattr(ctrl, "max_length", 8)}
    return (vae, ctrl), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_z", type=int, default=20, help="samples per control/seed")
    ap.add_argument("--n_seeds", type=int, default=5, help="layout seeds")
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="action-sampling temperature for the baselines")
    ap.add_argument("--styles", nargs="*", default=list(CANONICAL.keys()))
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    save_dir = os.path.join(HERE, "diversity_metrics")
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset …")
    dataset = MiniGridDataset(
        trajectory_paths=paths, sampling=True, index_channel_only=True,
        state_normalization_factor=1, action_normalization_factor=1,
        max_len=20, control_dim=CONTROL_DIM,
    )
    state_mean, state_std = dataset.state_mean, dataset.state_std

    # Overwrite CANONICAL placeholders with per-style dataset means so the
    # conditioning is data-accurate (keys stay the four-style names).
    ctrl_arr = np.asarray(dataset.controls)
    task_arr = np.asarray(dataset.tasks)
    for sid, sname in STYLE_ORDER.items():
        mask = task_arr == sid
        if mask.any():
            CANONICAL[sname] = ctrl_arr[mask].mean(axis=0).astype(np.float32)

    models, ctx = load_models()
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    model_names = ["StyleDT", "ControlDT"]

    table = {}          # table[style][model] = aggregated metrics
    overlay_data = {}   # overlay_data[style][model] = (paths, map_img) for seed[0]

    for style in args.styles:
        ctrl_vec = CANONICAL[style]
        table[style] = {}
        overlay_data[style] = {}
        print(f"\n=== {style.upper()}  c={np.round(ctrl_vec, 3)} ===")
        for name in model_names:
            per_seed = []
            for si, seed in enumerate(seeds):
                p, ach, img = collect_group(
                    name, models, ctrl_vec, style, state_mean, state_std,
                    seed, args.n_z, args.temperature, ctx[name],
                )
                per_seed.append(group_metrics(p, ach, style))
                if si == 0:
                    # Only plot style-consistent (successful) trajectories.
                    consistent_paths = [pp for pp, aa in zip(p, ach) if aa == style]
                    overlay_data[style][name] = (consistent_paths, img)
            agg = aggregate(per_seed)
            table[style][name] = agg
            print(f"  {name:10s}  D_eff={agg['D_traj_eff_mean']:.3f}±{agg['D_traj_eff_std']:.3f}"
                  f"   (D_traj={agg['D_traj_mean']:.3f} × SAR={agg['SAR_mean']:.3f})"
                  f"   H_eff={agg['H_route_eff_mean']:.3f}"
                  f"   H_route={agg['H_route_mean']:.3f}")

    # ---- save numbers ----
    with open(os.path.join(save_dir, "diversity_results.json"), "w") as f:
        json.dump({"config": vars(args), "table": table}, f, indent=2)
    print(f"\nSaved {os.path.join(save_dir, 'diversity_results.json')}")

    # ---- printed summary table ----
    print("\n" + "=" * 92)
    print(f"{'style':12s} {'model':10s} {'D_eff↑':>11s} {'D_traj':>9s} "
          f"{'H_eff↑':>9s} {'H_route':>9s} {'SAR↑':>9s}")
    print("(D_eff = D_traj × SAR ; H_eff = H_route × SAR)")
    print("-" * 92)
    for style in args.styles:
        for name in model_names:
            a = table[style][name]
            print(f"{style:12s} {name:10s} "
                  f"{a['D_traj_eff_mean']:6.3f}±{a['D_traj_eff_std']:.3f} "
                  f"{a['D_traj_mean']:9.3f} "
                  f"{a['H_route_eff_mean']:9.3f} "
                  f"{a['H_route_mean']:9.3f} "
                  f"{a['SAR_mean']:6.3f}±{a['SAR_std']:.3f}")
    print("=" * 92)

    # ---- plots ----
    print("\nPlots:")
    for style in args.styles:
        plot_overlay(style, overlay_data[style], save_dir, args.n_z)
    plot_scatter(table, save_dir)


if __name__ == "__main__":
    main()