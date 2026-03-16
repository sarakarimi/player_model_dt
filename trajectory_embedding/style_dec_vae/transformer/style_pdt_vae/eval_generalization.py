"""
Generalization evaluation for StyleVAE and ControlDT.

Tests two regimes:
  1. Interpolated controls  — linear blends between canonical style vectors
  2. OOD controls           — control vectors NOT seen during training (user-supplied)

For each control vector, N rollouts are run with no fixed target style so we
measure purely emergent behaviour.  We report:
  - Task performance  (success_rate, avg_return, detection_rate, episode_length)
  - Behavioural outcomes (enemy_distance, path_efficiency, weapon/camo usage)
  - Control adherence  (Spearman r between intended control dims and outcomes)

Usage:
    python trajectory_embedding/style_dec_vae/transformer/style_pdt_vae/eval_generalization.py
"""

import json
import os
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import shared infrastructure
# ---------------------------------------------------------------------------
from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths
from trajectory_embedding.style_dec_vae.transformer.style_pdt_vae.evaluate_metrics import (
    CONTROL_NAMES,
    CONTROL_OUTCOME_MAP,
    MAX_ENEMY_DIST,
    EpisodeRecord,
    MiniGridDataset,
    _rollout_episode,
    set_global_seeds,
)
from trajectory_embedding.style_dec_vae.transformer.style_pdt_vae.control_prompt_pdt import (
    ControlConditionedDT,
    CONTROL_DIM,
)
from trajectory_embedding.style_dec_vae.transformer.style_pdt_vae.pdt_vae_with_prior import (
    StyleVAEPromptDT,
)
from envs.three_style_env import MiniGridThreeStyles
from scipy.stats import spearmanr


HERE    = os.path.dirname(__file__)
DEVICE  = "cpu"
SEEDS   = [0, 1, 2, 3, 4]


# ===========================================================================
# >>> USER-SUPPLIED CONTROL VECTORS <<<
#
# Each entry is:   "label": np.array([risk_tolerance, resource_pref, commitment])
#
# Dimensions (all in [0, 1]):
#   risk_tolerance  — how close the agent gets to the enemy  (1 = risky/adjacent)
#   resource_pref   — item-pickup tendency  (0 = no items, 1 = always pick up)
#   commitment      — path directness  (1 = straight to goal)
# ===========================================================================

# ---------------------------------------------------------------------------
# Canonical style means from training data (mean of per-trajectory controls)
#   bypass:     risk=0.67  resource=0.03  commitment=0.81
#   weapon:     risk=0.92  resource=0.55  commitment=0.58
#   camouflage: risk=0.67  resource=0.53  commitment=0.77
#
# Each combination has 4 steps linearly interpolated between the two styles:
#   step 0 → 100% A /   0% B   (pure A, used as in-distribution anchor)
#   step 1 →  60% A /  40% B
#   step 2 →  40% A /  60% B
#   step 3 →   0% A / 100% B   (pure B, used as in-distribution anchor)
# ---------------------------------------------------------------------------

_B  = np.array([0.670, 0.030, 0.810], dtype=np.float32)   # bypass
_W  = np.array([0.920, 0.550, 0.580], dtype=np.float32)   # weapon
_C  = np.array([0.670, 0.530, 0.770], dtype=np.float32)   # camouflage

def _lerp(a, b, t):
    return ((1 - t) * a + t * b).astype(np.float32)

# {combo_name: [(step_label, control_vector), ...]}
STYLE_COMBINATIONS: Dict[str, list] = {
    "bypass_x_camouflage": [
        ("100% bypass\n0% camo",   _lerp(_B, _C, 0.0)),
        ("60% bypass\n40% camo",   _lerp(_B, _C, 0.4)),
        ("40% bypass\n60% camo",   _lerp(_B, _C, 0.5)),
        ("0% bypass\n100% camo",   _lerp(_B, _C, 1.0)),
    ],
    "weapon_x_camouflage": [
        ("100% weapon\n0% camo",   _lerp(_W, _C, 0.0)),
        ("60% weapon\n40% camo",   _lerp(_W, _C, 0.35)),
        ("40% weapon\n60% camo",   _lerp(_W, _C, 0.46)),
        ("0% weapon\n100% camo",   _lerp(_W, _C, 1.0)),
    ],
    "weapon_x_bypass": [
        ("100% weapon\n0% bypass", _lerp(_W, _B, 0.0)),
        ("60% weapon\n40% bypass", _lerp(_W, _B, 0.3)),
        ("40% weapon\n60% bypass", _lerp(_W, _B, 0.7)),
        ("0% weapon\n100% bypass", _lerp(_W, _B, 1.0)),
    ],
}

OOD_CONTROLS: Dict[str, np.ndarray] = {
    # --- paste your out-of-distribution vectors here ---
    "weapon_x_camouflage": np.array([0.82, 0.54, 0.67], dtype=np.float32),
    "camouflage_x_bypass": np.array([0.67, 0.19, 0.79], dtype=np.float32),
    "weapon_x_bypass": np.array([0.92, 0.03, 0.38], dtype=np.float32),

}

# Number of rollout episodes per control vector
NUM_EPISODES = 1 #20


# ===========================================================================
# Rollout helpers
# ===========================================================================

def _run_episodes_with_control(
    control_vec:  np.ndarray,
    model_name:   str,
    vae_model:    Optional[StyleVAEPromptDT],
    ctrl_model:   Optional[ControlConditionedDT],
    dataset:      MiniGridDataset,
    num_episodes: int,
    seed_offset:  int,
    max_ep_len:   int = 100,
    max_context:  int = 8,
    initial_rtg:  float = 1.0,
) -> List[EpisodeRecord]:
    """
    Run `num_episodes` rollouts using the given raw control vector.
    No target style is specified — emergent behaviour is recorded.
    Returns a list of EpisodeRecord with control_vector set.
    """
    state_mean = torch.tensor(dataset.state_mean, device=DEVICE, dtype=torch.float32)
    state_std  = torch.tensor(dataset.state_std,  device=DEVICE, dtype=torch.float32)
    ctrl_np    = np.asarray(control_vec, dtype=np.float32)

    # Build a one-off adapter-like object that holds the control conditioning
    if model_name == "StyleDT":
        assert vae_model is not None
        ctrl_tensor = torch.tensor(ctrl_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z           = vae_model.sample_z_from_prior(ctrl_tensor, deterministic=False)
        style_tokens = vae_model.latent_to_style_tokens(z)

        class _VAEAdHoc:
            def eval(self): vae_model.eval()
            def train(self): vae_model.train()
            state_mean = dataset.state_mean
            state_std  = dataset.state_std
            def get_action(_, states, actions, rtgs, timesteps, attn_mask):
                with torch.no_grad():
                    _, ap, _ = vae_model.dt(
                        states=states, actions=actions.squeeze(-1),
                        returns_to_go=rtgs, timesteps=timesteps,
                        attention_mask=attn_mask, style_tokens=style_tokens,
                    )
                return int(torch.argmax(ap[:, -1], dim=-1).item())
            def control_vector(_): return ctrl_np

        adapter = _VAEAdHoc()

    else:  # ControlDT
        assert ctrl_model is not None
        ctrl_tensor = torch.tensor(ctrl_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        class _CtrlAdHoc:
            def eval(self): ctrl_model.eval()
            def train(self): ctrl_model.train()
            state_mean = dataset.state_mean
            state_std  = dataset.state_std
            def get_action(_, states, actions, rtgs, timesteps, attn_mask):
                with torch.no_grad():
                    _, ap, _ = ctrl_model.forward(
                        states=states, actions=actions.squeeze(-1), rewards=None,
                        returns_to_go=rtgs, timesteps=timesteps,
                        controls=ctrl_tensor, attention_mask=attn_mask,
                    )
                return int(torch.argmax(ap[:, -1], dim=-1).item())
            def control_vector(_): return ctrl_np

        adapter = _CtrlAdHoc()

    adapter.eval()
    records: List[EpisodeRecord] = []

    with torch.no_grad():
        for ep in range(num_episodes):
            env = MiniGridThreeStyles(
                target_style=None,   # no target — purely measure emergent behaviour
                target_bonus=0.0,
                non_target_penalty=0.0,
                easy_env=False,
                agent_view_size=3,
                randomize_layout=True,
                # render_mode="human",
            )
            outcome = _rollout_episode(
                adapter, env, state_mean, state_std,
                DEVICE, initial_rtg, max_ep_len, max_context,
                seed=seed_offset + ep,
            )
            env.close()
            records.append(EpisodeRecord(
                style_id=0,          # unknown style, placeholder
                target_style="none",
                control_vector=ctrl_np.copy(),
                **outcome,
            ))

    adapter.train()
    return records


STYLE_LABELS   = ["bypass", "weapon", "camouflage"]
STYLE_COLORS   = {"bypass": "#4C72B0", "weapon": "#DD8452", "camouflage": "#55A868"}
UNKNOWN_COLOR  = "#999999"


def _aggregate_records(records: List[EpisodeRecord]) -> dict:
    """Compute performance, behavioural metrics, and style distribution."""
    n       = len(records)
    success = [r for r in records if r.success]
    n_suc   = len(success)

    def _mf(lst, field):
        vals = [getattr(r, field) for r in lst if getattr(r, field) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    # Style distribution across ALL episodes (including failures)
    style_counts = {s: 0 for s in STYLE_LABELS}
    style_counts["unknown"] = 0
    for r in records:
        style = r.achieved_style if r.achieved_style in STYLE_LABELS else "unknown"
        style_counts[style] += 1
    style_dist = {s: style_counts[s] / n for s in list(STYLE_LABELS) + ["unknown"]}

    # Shannon entropy of style distribution (higher = more diverse / uncertain)
    probs = np.array([style_dist[s] for s in STYLE_LABELS + ["unknown"]])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "n_episodes":            n,
        "success_rate":          n_suc / n,
        "avg_return":            float(np.mean([r.episode_return for r in records])),
        "avg_episode_length":    float(np.mean([r.length for r in records])),
        "detection_rate":        float(np.mean([r.detected for r in records])),
        "avg_enemy_distance":    _mf(success, "avg_enemy_distance"),
        "avg_path_efficiency":   _mf(success, "path_efficiency"),
        "weapon_usage_rate":     float(np.mean([r.picked_weapon      for r in success])) if success else float("nan"),
        "camouflage_usage_rate": float(np.mean([r.picked_camouflage  for r in success])) if success else float("nan"),
        "style_distribution":    style_dist,
        "style_entropy":         entropy,
    }


def _control_adherence(records: List[EpisodeRecord]) -> dict:
    """Spearman r between each control dim and its corresponding outcome."""
    suc = [r for r in records
           if r.success
           and r.control_vector is not None
           and r.avg_enemy_distance is not None
           and r.path_efficiency is not None]
    if len(suc) < 3:
        return {"mean_abs_spearman_r": float("nan")}

    outcomes = {
        "inv_min_dist":   np.array([1.0 - min(r.min_enemy_distance / MAX_ENEMY_DIST, 1.0) for r in suc]),
        "resource_used":  np.clip(np.array([r.items_picked / 2.0 for r in suc]), 0, 1),
        "path_efficiency": np.array([r.path_efficiency for r in suc]),
    }
    rs = {}
    for dim_idx, outcome_key, _ in CONTROL_OUTCOME_MAP:
        ctrl_vals = np.array([r.control_vector[dim_idx] for r in suc])
        r_val, p_val = spearmanr(ctrl_vals, outcomes[outcome_key])
        rs[CONTROL_NAMES[dim_idx]] = {"spearman_r": float(r_val), "p_value": float(p_val)}
    rs["mean_abs_spearman_r"] = float(np.mean([abs(v["spearman_r"]) for v in rs.values() if isinstance(v, dict)]))
    return rs


# ===========================================================================
# Per-model evaluation over all named control vectors + all seeds
# ===========================================================================

def evaluate_control_set(
    named_controls: Dict[str, np.ndarray],
    model_name:     str,
    vae_model:      Optional[StyleVAEPromptDT],
    ctrl_model:     Optional[ControlConditionedDT],
    dataset:        MiniGridDataset,
    seeds:          List[int] = SEEDS,
    num_episodes:   int = NUM_EPISODES,
    group_label:    str = "",
) -> Dict[str, dict]:
    """
    For each named control vector, run `num_episodes` rollouts across all seeds.
    Returns {control_label: {"performance": dict, "adherence": dict}}.
    """
    results = {}
    for label, ctrl in named_controls.items():
        print(f"  [{model_name}] {group_label} '{label}'  ctrl={np.round(ctrl, 3)}")
        all_records: List[EpisodeRecord] = []
        for seed in seeds:
            set_global_seeds(seed)
            recs = _run_episodes_with_control(
                control_vec=ctrl,
                model_name=model_name,
                vae_model=vae_model,
                ctrl_model=ctrl_model,
                dataset=dataset,
                num_episodes=num_episodes,
                seed_offset=seed * 10_000,
            )
            all_records.extend(recs)

        results[label] = {
            "intended_control": ctrl.tolist(),
            "performance":  _aggregate_records(all_records),
            "adherence":    _control_adherence(all_records),
        }
    return results


# ===========================================================================
# Printing
# ===========================================================================

PERF_KEYS = [
    ("success_rate",        "Success rate"),
    ("avg_return",          "Avg return"),
    ("avg_episode_length",  "Ep. length"),
    ("detection_rate",      "Detection rate"),
    ("avg_enemy_distance",  "Enemy distance"),
    ("avg_path_efficiency", "Path efficiency"),
    ("weapon_usage_rate",   "Weapon usage"),
    ("camouflage_usage_rate","Camo usage"),
]


def print_generalization_table(results: Dict[str, dict], model_name: str, group_label: str):
    labels = list(results.keys())
    bar = "=" * (30 + 14 * len(labels))
    print(f"\n{bar}")
    print(f"  {model_name} — {group_label}")
    print(bar)

    # Header
    row = f"{'metric':<28}"
    for lbl in labels:
        row += f"  {lbl[:12]:>12}"
    print(row)
    print("-" * len(row))

    print("[ Task performance ]")
    for key, name in PERF_KEYS:
        row = f"  {name:<26}"
        for lbl in labels:
            v = results[lbl]["performance"].get(key, float("nan"))
            row += f"  {v:>12.3f}" if not np.isnan(v) else f"  {'nan':>12}"
        print(row)

    print("[ Control adherence (Spearman r) ]")
    for dim in CONTROL_NAMES:
        row = f"  {dim:<26}"
        for lbl in labels:
            v = results[lbl]["adherence"].get(dim, {}).get("spearman_r", float("nan"))
            row += f"  {v:>+12.3f}" if not np.isnan(v) else f"  {'nan':>12}"
        print(row)

    row = f"  {'mean |r|':<26}"
    for lbl in labels:
        v = results[lbl]["adherence"].get("mean_abs_spearman_r", float("nan"))
        row += f"  {v:>12.3f}" if not np.isnan(v) else f"  {'nan':>12}"
    print(row)
    print(bar)


def plot_style_distributions(
    group_data: Dict[str, Dict[str, dict]],
    group_name: str,
    save_dir:   str,
):
    """
    One figure per control vector.  Each figure has one grouped-bar cluster
    per model, showing the achieved-style distribution as a stacked bar plus
    a style-entropy annotation.

    group_data: {model_name: {control_label: result_dict}}
    """
    model_names   = [mn for mn, res in group_data.items() if res]
    if not model_names:
        return

    # Collect the union of control labels across models
    all_labels = []
    for mn in model_names:
        for lbl in group_data[mn]:
            if lbl not in all_labels:
                all_labels.append(lbl)

    all_styles = STYLE_LABELS + ["unknown"]

    for ctrl_label in all_labels:
        fig, axes = plt.subplots(1, len(model_names),
                                 figsize=(4.5 * len(model_names), 5),
                                 sharey=True)
        if len(model_names) == 1:
            axes = [axes]

        fig.suptitle(
            f"Achieved style distribution\n"
            f"Control: {ctrl_label}  [{group_name}]",
            fontsize=12, fontweight="bold",
        )

        for ax, mn in zip(axes, model_names):
            res = group_data[mn].get(ctrl_label)
            if res is None:
                ax.set_title(mn)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            dist    = res["performance"]["style_distribution"]
            entropy = res["performance"]["style_entropy"]

            # Build stacked bar (x = single bar per model)
            bottom = 0.0
            for style in all_styles:
                proportion = dist.get(style, 0.0)
                color = STYLE_COLORS.get(style, UNKNOWN_COLOR)
                ax.bar(0, proportion, bottom=bottom,
                       color=color, alpha=0.88,
                       label=style if ax == axes[0] else "_nolegend_",
                       width=0.5)
                if proportion > 0.04:
                    ax.text(0, bottom + proportion / 2,
                            f"{proportion:.0%}", ha="center", va="center",
                            fontsize=16, fontweight="bold", color="white")
                bottom += proportion

            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([])
            # ax.set_title(f"{mn}\nH={entropy:.2f} bits", fontsize=10)
            ax.set_ylabel("Proportion of episodes", fontsize=16)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        # Single shared legend
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=STYLE_COLORS.get(s, UNKNOWN_COLOR), alpha=0.88)
                   for s in all_styles]
        fig.legend(handles, all_styles, loc="lower center",
                   ncol=len(all_styles), fontsize=16,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        safe_label = ctrl_label.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(save_dir, "plots",
                                f"style_dist_{group_name}_{safe_label}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close()


# ===========================================================================
# Style-combination evaluation + plotting
# ===========================================================================

def evaluate_style_combinations(
    combinations:  Dict[str, list],
    model_name:    str,
    vae_model:     Optional[StyleVAEPromptDT],
    ctrl_model:    Optional[ControlConditionedDT],
    dataset:       MiniGridDataset,
    seeds:         List[int] = SEEDS,
    num_episodes:  int = NUM_EPISODES,
) -> Dict[str, Dict[str, dict]]:
    """
    For every combination group (e.g. "bypass_x_camouflage") run each of the
    4 interpolation steps across all seeds.

    Returns:
        {combo_name: {step_label: result_dict}}
    where result_dict = {"intended_control", "performance", "adherence"}.
    """
    combo_results: Dict[str, Dict[str, dict]] = {}
    for combo_name, steps in combinations.items():
        print(f"  [{model_name}] combination: {combo_name}")
        step_results: Dict[str, dict] = {}
        for step_label, ctrl_vec in steps:
            clean_label = step_label.replace("\n", " ")
            print(f"    step '{clean_label}'  ctrl={np.round(ctrl_vec, 3)}")
            all_records: List[EpisodeRecord] = []
            for seed in seeds:
                set_global_seeds(seed)
                recs = _run_episodes_with_control(
                    control_vec=ctrl_vec,
                    model_name=model_name,
                    vae_model=vae_model,
                    ctrl_model=ctrl_model,
                    dataset=dataset,
                    num_episodes=num_episodes,
                    seed_offset=seed * 10_000,
                )
                all_records.extend(recs)
            step_results[step_label] = {
                "intended_control": ctrl_vec.tolist(),
                "performance":      _aggregate_records(all_records),
                "adherence":        _control_adherence(all_records),
            }
        combo_results[combo_name] = step_results
    return combo_results


def plot_combination_transitions(
    combo_data: Dict[str, Dict[str, Dict[str, dict]]],
    save_dir:   str,
):
    """
    One figure per combination pair.  Each figure has one subplot column per
    model (StyleVAE | ControlDT) — both share the same y axis.

    Each subplot shows 4 stacked bars (one per interpolation step) with the
    achieved style proportion and an entropy annotation per bar.

    combo_data: {combo_name: {model_name: {step_label: result_dict}}}
    """
    all_styles = STYLE_LABELS + ["unknown"]
    style_colors = [STYLE_COLORS.get(s, UNKNOWN_COLOR) for s in all_styles]

    for combo_name, model_dict in combo_data.items():
        model_names = [mn for mn in model_dict if model_dict[mn]]
        if not model_names:
            continue

        n_models = len(model_names)
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(5.5 * n_models, 5),
            sharey=True,
        )
        if n_models == 1:
            axes = [axes]

        # combo_display = combo_name.replace("_x_", " ↔ ").replace("_", " ")
        # fig.suptitle(
        #     f"Style-interpolation transitions: {combo_display}",
        #     fontsize=13, fontweight="bold",
        # )

        for ax, mn in zip(axes, model_names):
            step_dict = model_dict[mn]
            if not step_dict:
                ax.set_title(mn)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            step_labels = list(step_dict.keys())
            x_positions = np.arange(len(step_labels))

            for xi, step_label in enumerate(step_labels):
                res  = step_dict[step_label]
                dist = res["performance"]["style_distribution"]
                H    = res["performance"]["style_entropy"]

                bottom = 0.0
                for style, color in zip(all_styles, style_colors):
                    p = dist.get(style, 0.0)
                    ax.bar(xi, p, bottom=bottom, color=color, alpha=0.88,
                           width=0.7,
                           label=style if xi == 0 else "_nolegend_")
                    if p > 0.05:
                        ax.text(xi, bottom + p / 2,
                                f"{p:.0%}", ha="center", va="center",
                                fontsize=16, fontweight="bold", color="white")
                    bottom += p

                # Entropy annotation above bar
                # ax.text(xi, 1.02, f"H={H:.2f}", ha="center", va="bottom",
                #         fontsize=8, color="#333333")

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [lbl.replace("\n", "\n") for lbl in step_labels],
                fontsize=11,
            )
            ax.set_ylim(0, 1.12)
            ax.set_title(mn, fontsize=16)
            ax.set_ylabel("Proportion of episodes", fontsize=16)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

        # Shared legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88)
                   for c in style_colors]
        fig.legend(handles, all_styles, loc="lower center",
                   ncol=len(all_styles), fontsize=13,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        out_path = os.path.join(save_dir, "plots",
                                f"style_interp_{combo_name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    control_dim = CONTROL_DIM
    dataset_params = {
        "sampling":                   True,
        "index_channel_only":         True,
        "state_normalization_factor":  1,
        "action_normalization_factor": 1,
        "max_len":                    20,
        "control_dim":                control_dim,
    }
    print("Loading dataset …")
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading StyleVAE …")
    vae_model = StyleVAEPromptDT(
        state_dim=9, act_dim=7, hidden_size=128, latent_dim=16,
        max_length=20, max_ep_len=100, action_tanh=False,
        beta=0.0085, control_dim=control_dim, prior_hidden=128,
        free_bits=0.0, n_layer=4, n_head=8,
    )
    vae_ckpt = os.path.join(HERE, "trained_models/style_prompt_dt_minigrid_controls_condprior.pth")
    if os.path.exists(vae_ckpt):
        vae_model.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
    else:
        print(f"  WARNING: VAE checkpoint not found at {vae_ckpt} — using random weights.")
    vae_model.to(DEVICE).eval()

    print("Loading ControlDT …")
    ctrl_model = ControlConditionedDT(
        state_dim=9, act_dim=7, hidden_size=128,
        control_dim=control_dim, max_length=8, max_ep_len=100,
        action_tanh=False, n_layer=4, n_head=8,
    )
    ctrl_ckpt = os.path.join(HERE, "trained_models/control_dt_minigrid.pth")
    if os.path.exists(ctrl_ckpt):
        ctrl_model.load_state_dict(torch.load(ctrl_ckpt, map_location=DEVICE))
    else:
        print(f"  WARNING: ControlDT checkpoint not found at {ctrl_ckpt} — using random weights.")
    ctrl_model.to(DEVICE).eval()

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    models = {"StyleDT": (vae_model, None), "ControlDT": (None, ctrl_model)}

    # combo_results[model_name][combo_name][step_label] = result_dict
    combo_results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    ood_results:   Dict[str, dict] = {}

    for model_name, (vae, ctrl) in models.items():
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

        # --- style-interpolation combinations ---
        print("-- Style-combination transitions --")
        combo_results[model_name] = evaluate_style_combinations(
            STYLE_COMBINATIONS, model_name, vae, ctrl, dataset,
        )
        for combo_name, step_dict in combo_results[model_name].items():
            print_generalization_table(step_dict, model_name, combo_name)

        # --- OOD controls ---
        if OOD_CONTROLS:
            print("-- OOD controls --")
            ood_results[model_name] = evaluate_control_set(
                OOD_CONTROLS, model_name, vae, ctrl, dataset,
                group_label="OOD",
            )
            print_generalization_table(
                ood_results[model_name], model_name, "OOD controls"
            )
        else:
            print("  (no OOD controls defined — add them to OOD_CONTROLS)")
            ood_results[model_name] = {}

    # ------------------------------------------------------------------
    # Plots + JSON
    # ------------------------------------------------------------------

    # Re-arrange combo_results to {combo_name: {model_name: step_dict}}
    # so plot_combination_transitions can iterate combos as outer loop
    per_combo: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for mn, combos in combo_results.items():
        for combo_name, step_dict in combos.items():
            per_combo.setdefault(combo_name, {})[mn] = step_dict

    plot_combination_transitions(per_combo, save_dir=HERE)

    if any(ood_results.values()):
        plot_style_distributions(ood_results, group_name="ood", save_dir=HERE)

    save_path = os.path.join(HERE, "eval_generalization_results.json")
    with open(save_path, "w") as f:
        json.dump(
            {"style_combinations": combo_results, "ood": ood_results},
            f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x),
        )
    print(f"\nSaved results to {save_path}")