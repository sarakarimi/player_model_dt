"""
Plot the distribution of per-trajectory control vectors by style.

Produces two figures saved alongside this script:
  controls_kde.png   – KDE of each control dim, one curve per style
  controls_heatmap.png – mean ± std heatmap (styles × dims)

Run from the repo root:
    python trajectory_embedding/style_dec_vae/transformer/style_pdt_vae/plot_controls.py
"""

import os
import sys

# Add repo root to path so all project modules resolve correctly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from style_decision_transformer import paths
from style_decision_transformer.transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTROL_NAMES = [
    "risk_tolerance",
    "resource_pref",
    # "stealth_pref",
    # "safety_pref",
    "commitment",
]
STYLE_NAMES = {0: "bypass", 1: "weapon", 2: "camouflage"}
STYLE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}  # blue, orange, green

SAVE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

dataset_params = {
    "sampling": True,
    "index_channel_only": True,
    "state_normalization_factor": 1,
    "action_normalization_factor": 1,
    "max_len": 8,
    "control_dim": len(CONTROL_NAMES),
}

print("Loading dataset …")
dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)

controls = dataset.controls          # [N, 5]  float32
tasks    = np.array(dataset.tasks)   # [N]     int

# Diagnostic: how many trajectories used episode_summary vs fallback
n_with_summary = sum(
    1 for ep in dataset.infos
    if isinstance(ep, dict) and ep.get("episode_summary") is not None
)
print(
    f"Trajectories with episode_summary : {n_with_summary}/{len(dataset.infos)}"
    f"  (rest use style-label fallback)"
)

# Collect per-style controls
style_controls = {
    sid: controls[tasks == sid] for sid in STYLE_NAMES
}
for sid, name in STYLE_NAMES.items():
    print(f"  {name}: {len(style_controls[sid])} trajectories")

# ---------------------------------------------------------------------------
# Figure 1 – KDE distributions
# ---------------------------------------------------------------------------

n_dims = len(CONTROL_NAMES)
fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4), sharey=False)
fig.suptitle("Control Vector Distributions by Style", fontsize=14, fontweight="bold")

for col, (dim_idx, dim_name) in enumerate(enumerate(CONTROL_NAMES)):
    ax = axes[col]
    for sid, name in STYLE_NAMES.items():
        data = style_controls[sid][:, dim_idx]
        sns.kdeplot(
            data,
            ax=ax,
            label=name,
            color=STYLE_COLORS[sid],
            linewidth=2,
            fill=True,
            alpha=0.25,
            clip=(0.0, 1.0),
        )
        # vertical line at the mean
        ax.axvline(
            data.mean(),
            color=STYLE_COLORS[sid],
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )

    ax.set_title(dim_name, fontsize=11)
    ax.set_xlabel("value [0–1]", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylabel("density" if col == 0 else "")
    ax.tick_params(labelsize=8)
    if col == n_dims - 1:
        ax.legend(title="style", fontsize=8, title_fontsize=8)

plt.tight_layout()
kde_path = os.path.join(SAVE_DIR, "plots/controls_kde.png")
plt.savefig(kde_path, dpi=150)
print(f"Saved {kde_path}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 – Mean ± std heatmap + error bars
# ---------------------------------------------------------------------------

n_styles = len(STYLE_NAMES)
means = np.zeros((n_styles, n_dims))
stds  = np.zeros((n_styles, n_dims))

for row, sid in enumerate(STYLE_NAMES):
    means[row] = style_controls[sid].mean(axis=0)
    stds[row]  = style_controls[sid].std(axis=0)

fig2 = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=fig2)
fig2.suptitle("Control Vector Summary by Style", fontsize=14, fontweight="bold")

# -- left: heatmap of means --
ax_heat = fig2.add_subplot(gs[0])
im = ax_heat.imshow(means, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
ax_heat.set_xticks(range(n_dims))
ax_heat.set_xticklabels(CONTROL_NAMES, rotation=30, ha="right", fontsize=9)
ax_heat.set_yticks(range(n_styles))
ax_heat.set_yticklabels([STYLE_NAMES[s] for s in STYLE_NAMES], fontsize=10)
ax_heat.set_title("Mean control value", fontsize=11)
for r in range(n_styles):
    for c in range(n_dims):
        ax_heat.text(
            c, r,
            f"{means[r, c]:.2f}\n±{stds[r, c]:.2f}",
            ha="center", va="center", fontsize=7.5,
            color="black" if means[r, c] < 0.65 else "white",
        )
plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04)

# -- right: grouped bar chart (mean ± std) --
ax_bar = fig2.add_subplot(gs[1])
x = np.arange(n_dims)
width = 0.25
for row, sid in enumerate(STYLE_NAMES):
    offset = (row - 1) * width
    bars = ax_bar.bar(
        x + offset, means[row], width,
        yerr=stds[row],
        label=STYLE_NAMES[sid],
        color=STYLE_COLORS[sid],
        alpha=0.85,
        capsize=3,
        error_kw={"linewidth": 1},
    )
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(
    [n.replace("_", "\n") for n in CONTROL_NAMES],
    fontsize=8,
)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_ylabel("mean ± std", fontsize=9)
ax_bar.set_title("Mean ± std per style", fontsize=11)
ax_bar.legend(fontsize=8)
ax_bar.yaxis.grid(True, alpha=0.4)
ax_bar.set_axisbelow(True)

plt.tight_layout()
heatmap_path = os.path.join(SAVE_DIR, "plots/controls_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
print(f"Saved {heatmap_path}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 – Violin plots (distribution shape + spread)
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4), sharey=True)
fig3.suptitle("Control Vector Spread by Style (violin)", fontsize=14, fontweight="bold")

for col, dim_name in enumerate(CONTROL_NAMES):
    ax = axes3[col]
    data_per_style = [style_controls[sid][:, col] for sid in STYLE_NAMES]
    parts = ax.violinplot(
        data_per_style,
        positions=range(n_styles),
        showmedians=True,
        showextrema=True,
    )
    # color each violin body
    for pc, sid in zip(parts["bodies"], STYLE_NAMES):
        pc.set_facecolor(STYLE_COLORS[sid])
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)

    ax.set_xticks(range(n_styles))
    ax.set_xticklabels([STYLE_NAMES[s] for s in STYLE_NAMES], fontsize=8, rotation=15)
    ax.set_title(dim_name, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("value [0–1]" if col == 0 else "")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
violin_path = os.path.join(SAVE_DIR, "plots/controls_violin.png")
plt.savefig(violin_path, dpi=150)
print(f"Saved {violin_path}")
plt.close()

print("Done.")