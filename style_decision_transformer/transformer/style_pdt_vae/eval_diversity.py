"""
Trajectory diversity visualisation for StyleDT.

For each canonical style, fixes one environment layout, samples N_Z
latent vectors z ~ p(z|c), runs one rollout per z on that layout, and
draws all paths overlaid on the rendered map.
ControlDT is shown alongside for comparison (same fixed env, same policy
repeated N_Z times — diversity comes only from env randomness; since the
seed is fixed the layout is the same but the agent starts deterministically,
so all paths are nearly identical, which is the point).

Usage:
    python trajectory_embedding/style_dec_vae/transformer/style_pdt_vae/eval_diversity.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)

from style_decision_transformer import paths
from style_decision_transformer.transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
    StyleVAEPromptDT,
)
from style_decision_transformer.transformer.style_pdt_vae.control_prompt_pdt import (
    ControlConditionedDT,
    CONTROL_DIM,
)
from envs.three_style_env import MiniGridThreeStyles

HERE   = os.path.dirname(__file__)
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Canonical control vectors (per-style training-data means)
# ---------------------------------------------------------------------------
CANONICAL = {
    "bypass":     np.array([0.670, 0.030, 0.810], dtype=np.float32),
    "weapon":     np.array([0.920, 0.550, 0.580], dtype=np.float32),
    "camouflage": np.array([0.670, 0.530, 0.770], dtype=np.float32),
}

N_Z         = 10     # distinct z samples per control vector
MAP_SEED    = 42     # fixed env seed — same layout for every trajectory
MAX_EP_LEN  = 100
MAX_CONTEXT = 8
INITIAL_RTG = 1.0


# ---------------------------------------------------------------------------
# Single rollout — returns agent (x, y) path and a rendered map image
# ---------------------------------------------------------------------------

def _rollout_positions(get_action_fn, state_mean, state_std, env_seed):
    """
    Run one episode and return:
      positions : list of (x, y) agent grid positions at each step
      map_img   : RGB render of the environment (captured after reset)
    """
    env = MiniGridThreeStyles(
        target_style=None, target_bonus=1.0, non_target_penalty=-1.0,
        easy_env=False, agent_view_size=3, randomize_layout=True,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=env_seed)
    map_img = env.render()   # capture layout before agent moves

    sm = torch.tensor(state_mean, dtype=torch.float32)
    ss = torch.tensor(state_std,  dtype=torch.float32)

    state     = torch.from_numpy(obs["image"][:, :, 0].flatten()).float()
    state     = (state - sm) / ss
    states    = state.reshape(1, 1, -1)
    actions   = torch.zeros((1, 1, 1), dtype=torch.long)
    rtgs      = torch.tensor([[[INITIAL_RTG]]], dtype=torch.float32)
    timesteps = torch.tensor([[0]], dtype=torch.long)

    positions = [tuple(env.agent_pos)]
    done = False
    t    = 0

    while not done and t < MAX_EP_LEN:
        attn   = torch.ones((1, states.shape[1]), dtype=torch.float32)
        action = get_action_fn(states, actions, rtgs, timesteps, attn)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t   += 1
        positions.append(tuple(env.agent_pos))

        if not done:
            ns = torch.from_numpy(obs["image"][:, :, 0].flatten()).float()
            ns = (ns - sm) / ss
            states    = torch.cat([states,    ns.reshape(1, 1, -1)], dim=1)
            actions   = torch.cat([actions,   torch.tensor([[[action]]], dtype=torch.long)], dim=1)
            rtgs      = torch.cat([rtgs,      rtgs[:, -1:] - reward], dim=1)
            timesteps = torch.cat([timesteps, torch.tensor([[t]], dtype=torch.long)], dim=1)
            if states.shape[1] > MAX_CONTEXT:
                states    = states[:, -MAX_CONTEXT:]
                actions   = actions[:, -MAX_CONTEXT:]
                rtgs      = rtgs[:, -MAX_CONTEXT:]
                timesteps = timesteps[:, -MAX_CONTEXT:]

    env.close()
    return positions, map_img


# ---------------------------------------------------------------------------
# Collect N_Z trajectories for one model / one control vector
# ---------------------------------------------------------------------------

def collect_trajectories(model_name, ctrl_vec, vae_model, ctrl_model,
                          state_mean, state_std, n_z=N_Z):
    """
    Returns (list_of_position_lists, map_image).

    StyleDT  : each of the n_z episodes uses a freshly sampled z ~ p(z|c).
    ControlDT: same deterministic policy repeated n_z times on the fixed map.
    """
    all_paths = []
    map_img   = None
    ctrl_t    = torch.tensor(ctrl_vec, dtype=torch.float32).unsqueeze(0)

    for i in range(n_z):
        if model_name == "StyleDT":
            with torch.no_grad():
                z            = vae_model.sample_z_from_prior(ctrl_t, deterministic=False)
                style_tokens = vae_model.latent_to_style_tokens(z)

            def get_action(states, actions, rtgs, timesteps, attn,
                           _st=style_tokens):
                with torch.no_grad():
                    _, ap, _ = vae_model.dt(
                        states=states, actions=actions.squeeze(-1),
                        returns_to_go=rtgs, timesteps=timesteps,
                        attention_mask=attn, style_tokens=_st,
                    )
                return int(torch.argmax(ap[:, -1], dim=-1).item())

        else:  # ControlDT
            def get_action(states, actions, rtgs, timesteps, attn,
                           _ct=ctrl_t):
                with torch.no_grad():
                    _, ap, _ = ctrl_model.forward(
                        states=states, actions=actions.squeeze(-1), rewards=None,
                        returns_to_go=rtgs, timesteps=timesteps,
                        controls=_ct, attention_mask=attn,
                    )
                return int(torch.argmax(ap[:, -1], dim=-1).item())

        positions, img = _rollout_positions(
            get_action, state_mean, state_std, env_seed=MAP_SEED,
        )
        all_paths.append(positions)
        if map_img is None:
            map_img = img   # use the first render as the shared background

    return all_paths, map_img


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_trajectory_diversity(results: dict, save_dir: str):
    """
    results: {style: {model_name: (list_of_paths, map_img)}}

    Produces one figure per style with one subplot per model.
    Each subplot overlays all N_Z trajectory paths on the rendered map.
    """
    models = list(next(iter(results.values())).keys())
    cmap   = plt.cm.plasma

    for style, model_dict in results.items():
        fig, axes = plt.subplots(
            1, len(models),
            figsize=(5 * len(models), 5),
            squeeze=False,
        )
        axes = axes[0]
        fig.suptitle(
            f"Trajectory diversity — {style.capitalize()} style  "
            f"(fixed layout, {N_Z} samples of $z \\sim p_\\psi(z \\mid c)$)",
            fontsize=12, fontweight="bold",
        )

        for ax, model_name in zip(axes, models):
            paths, map_img = model_dict[model_name]

            if map_img is not None:
                ax.imshow(map_img, origin="upper")
                h, w = map_img.shape[:2]
                cell   = w / 12.0   # 9 interior + 2 border columns
                offset = cell       # shift paths by one border cell
            else:
                h, w, cell, offset = 450, 450, 45.0, 45.0
                ax.set_facecolor("#e8e8e8")

            n = len(paths)
            colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

            for traj, color in zip(paths, colors):
                if len(traj) < 2:
                    continue
                xs = [offset + (x + 0.5) * cell for x, y in traj]
                ys = [offset + (y + 0.5) * cell for x, y in traj]
                ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.75)
                ax.plot(xs[0],  ys[0],  "o", color=color, markersize=5,  zorder=5)
                ax.plot(xs[-1], ys[-1], "*", color=color, markersize=9, zorder=5)

            ax.set_title(model_name, fontsize=11)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)   # image origin is top-left
            ax.axis("off")

        sm_cb = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=1, vmax=n)
        )
        sm_cb.set_array([])
        fig.colorbar(sm_cb, ax=axes[-1], shrink=0.65,
                     label="Trajectory index")

        plt.tight_layout()
        out = os.path.join(save_dir, "plots", f"traj_diversity_{style}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset_params = dict(
        sampling=True,
        index_channel_only=True,
        state_normalization_factor=1,
        action_normalization_factor=1,
        max_len=20,
        control_dim=CONTROL_DIM,
    )
    print("Loading dataset …")
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)

    print("Loading StyleDT …")
    vae_model = StyleVAEPromptDT(
        state_dim=9, act_dim=7, hidden_size=128, latent_dim=16,
        max_length=20, max_ep_len=100, action_tanh=False,
        beta=0.0085, control_dim=CONTROL_DIM, prior_hidden=128,
        free_bits=0.0, n_layer=4, n_head=8,
    )
    vae_ckpt = os.path.join(HERE, "trained_models/style_prompt_dt_minigrid_controls_condprior.pth")
    if os.path.exists(vae_ckpt):
        vae_model.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
        print(f"  Loaded {vae_ckpt}")
    else:
        print(f"  WARNING: checkpoint not found at {vae_ckpt} — using random weights.")
    vae_model.to(DEVICE).eval()

    print("Loading ControlDT …")
    ctrl_model = ControlConditionedDT(
        state_dim=9, act_dim=7, hidden_size=128,
        control_dim=CONTROL_DIM, max_length=8, max_ep_len=100,
        action_tanh=False, n_layer=4, n_head=8,
    )
    ctrl_ckpt = os.path.join(HERE, "trained_models/control_dt_minigrid.pth")
    if os.path.exists(ctrl_ckpt):
        ctrl_model.load_state_dict(torch.load(ctrl_ckpt, map_location=DEVICE))
        print(f"  Loaded {ctrl_ckpt}")
    else:
        print(f"  WARNING: checkpoint not found at {ctrl_ckpt} — using random weights.")
    ctrl_model.to(DEVICE).eval()

    state_mean = dataset.state_mean
    state_std  = dataset.state_std

    results = {}
    for style, ctrl_vec in CANONICAL.items():
        print(f"\n=== {style.upper()} (c={np.round(ctrl_vec, 3)}) ===")
        results[style] = {}
        for model_name, (vae, ctrl) in [
            ("StyleDT",   (vae_model,  None)),
            ("ControlDT", (None, ctrl_model)),
        ]:
            print(f"  {model_name}: collecting {N_Z} trajectories …")
            paths, map_img = collect_trajectories(
                model_name, ctrl_vec, vae, ctrl, state_mean, state_std,
            )
            results[style][model_name] = (paths, map_img)

    plot_trajectory_diversity(results, save_dir=HERE)