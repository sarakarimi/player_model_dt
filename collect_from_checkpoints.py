"""
Run greedy inference with each saved PPO checkpoint, render the rollouts live so
you can watch them, and save the collected trajectories as datasets (one .gz per
checkpoint) in the exact format `MiniGridDataset` reads.

Self-contained: it does NOT use collect_trajectories / agent.py / vector envs.
It rebuilds the small actor MLP straight from the checkpoint weights and drives a
single MiniGridMultiStyles env with render_mode="human".

Run it yourself, e.g.:
    python collect_from_checkpoints.py
    python collect_from_checkpoints.py --episodes 50 --delay 0.02
    python collect_from_checkpoints.py --no-render --episodes 300      # bulk, no window
    python collect_from_checkpoints.py \
        --checkpoints "trained_models/multi_style_env_hard_bypass_*_PPO.pt"

Each .gz contains observations/actions/rewards/dones/truncated and `infos`. On the
terminal step of every episode the info is stored as {"final_info": [info]} where
`info` carries the env's `episode_summary` — which is what
`multi_style_controls_from_episode_summary` turns into the
[risk_taking, stealth_exposure, route_directness] controls.
"""
import argparse
import glob
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# repo root on path so `envs` is importable no matter where this is launched from
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.multi_style_env import MiniGridMultiStyles  # noqa: E402


def build_actor(model_state_dict):
    """Rebuild the FC actor (Flatten -> Linear -> Tanh -> Linear -> Tanh -> Linear)
    from the checkpoint weights. Dims are inferred from the weight shapes."""
    sd = {k[len("actor."):]: v for k, v in model_state_dict.items()
          if k.startswith("actor.")}
    in_dim = sd["1.weight"].shape[1]
    hidden = sd["1.weight"].shape[0]
    out_dim = sd["5.weight"].shape[0]
    actor = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def make_env(env_cfg, render):
    """Build a single MiniGridMultiStyles env mirroring ppo/make_env.py:33,
    using the checkpoint's stored environment_config."""
    return MiniGridMultiStyles(
        target_style=env_cfg["env_mode"],
        target_bonus=1.0,
        non_target_penalty=-1.0,
        free_item_placement=True,
        bypass_corridor=env_cfg.get("bypass_corridor"),
        agent_view_size=env_cfg.get("view_size", 3),
        max_steps=env_cfg.get("max_steps", 100),
        render_mode="human" if render else None,
    )


@torch.no_grad()
def collect_checkpoint(ckpt_path, out_path, episodes, render, delay, seed):
    saved = torch.load(ckpt_path, map_location="cpu")
    env_cfg = json.loads(saved["environment_config"])

    # Skip untrained checkpoints: a freshly-initialised policy has a near-zero
    # final layer (std ~ the 0.01 init), produces ~uniform logits, and never
    # reaches the goal -> no episode_summary, so its controls would be garbage.
    final_std = float(saved["model_state_dict"]["actor.5.weight"].std())
    # if final_std < 0.01:
    #     print(f"    SKIP (looks untrained: actor final-layer std={final_std:.4f})")
    #     return None

    actor = build_actor(saved["model_state_dict"])
    env = make_env(env_cfg, render)

    # per-step buffers; each entry is batched as [num_envs=1, ...] to match the
    # [T, B, ...] layout MiniGridDataset expects.
    observations, actions, rewards, dones, truncated, infos = [], [], [], [], [], []
    ep_returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if render:
            env.render()
        done = False
        ep_ret = 0.0
        while not done:
            img = np.asarray(obs["image"], dtype=np.float32)          # (H, W, C)
            logits = actor(torch.from_numpy(img).unsqueeze(0))        # [1, n_actions]
            action = int(torch.argmax(logits, dim=-1).item())         # greedy

            next_obs, reward, terminated, trunc, info = env.step(action)
            done = bool(terminated or trunc)

            observations.append([img])
            actions.append([action])
            rewards.append([float(reward)])
            dones.append([bool(terminated)])
            truncated.append([bool(trunc)])
            # episode_summary lives in `info` on the terminal step; store it the
            # way the dataset reads it: infos[t]["final_info"][b].
            infos.append({"final_info": [info]} if done else {})

            if render:
                env.render()
                if delay:
                    time.sleep(delay)

            obs = next_obs
            ep_ret += float(reward)
        ep_returns.append(ep_ret)
        print(f"    ep {ep + 1}/{episodes}  return={ep_ret:.3f}  steps={len(observations)}", end="\r")

    env.close()

    data = {
        "observations": np.array(observations, dtype=float),   # [T, 1, H, W, C]
        "actions": np.array(actions, dtype=np.int64),          # [T, 1]
        "rewards": np.array(rewards, dtype=float),             # [T, 1]
        "dones": np.array(dones, dtype=bool),                  # [T, 1]
        "truncated": np.array(truncated, dtype=bool),          # [T, 1]
        "rtgs": np.array([], dtype=float),
        "infos": np.array(infos, dtype=object),                # [T]
    }
    metadata = {"args": json.dumps(env_cfg), "time": time.time()}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump({"data": data, "metadata": metadata}, f)

    mb = os.path.getsize(out_path) / 1e6
    print(f"\n    saved {len(ep_returns)} episodes -> {out_path} ({mb:.1f} MB) "
          f"| mean return {np.mean(ep_returns):.3f}")
    return float(np.mean(ep_returns))


def parse_args():
    p = argparse.ArgumentParser(description="Greedy inference + render + save datasets per PPO checkpoint.")
    p.add_argument("--checkpoints", default=os.path.join(REPO_ROOT, "trained_models", "multi_style_env_hard_*_PPO.pt"),
                   help="Glob for the PPO checkpoint files.")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "datasets", "minigrid", "collected_from_checkpoints"),
                   help="Where to write one <checkpoint>.gz per checkpoint.")
    p.add_argument("--episodes", type=int, default=5, help="Episodes to roll out per checkpoint.")
    p.add_argument("--delay", type=float, default=0.03, help="Seconds to sleep between rendered frames.")
    p.add_argument("--seed", type=int, default=42, help="Base seed (episode i uses seed+i).")
    p.add_argument("--no-render", dest="render", action="store_false", help="Disable the live window (faster).")
    p.add_argument("--overwrite", action="store_true", help="Re-collect even if the .gz exists.")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoints = sorted(glob.glob(args.checkpoints))
    if not checkpoints:
        print(f"No checkpoints matched: {args.checkpoints}")
        sys.exit(1)

    print(f"{len(checkpoints)} checkpoint(s) -> {args.out_dir}")
    for i, ckpt in enumerate(checkpoints, 1):
        name = os.path.basename(ckpt).replace("_PPO.pt", "")
        out_path = os.path.join(args.out_dir, f"{name}.gz")
        print(f"\n[{i}/{len(checkpoints)}] {name}")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"    skip (exists). use --overwrite to redo.")
            continue
        collect_checkpoint(ckpt, out_path, args.episodes, args.render, args.delay, args.seed)


if __name__ == "__main__":
    main()