"""
Play saved PPO checkpoints with on-screen rendering.

Loads every top-level checkpoint in `trained_models/*.pt`, rebuilds the FCAgent
actor from the saved weights, and rolls it out in MiniGridMultiStyles with
`render_mode="human"` for a fixed number of rounds per checkpoint.

The actor (and the way observations are fed to it) mirrors ppo/agent.py exactly:
the raw 7x7x3 image is cast to float32 and flattened to 147 dims, and actions are
*sampled* from the policy's Categorical (PPO is stochastic), which is what the
trajectories were collected with.

Run:  python -m ppo.play_checkpoints
"""

import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from envs.multi_style_minigrid_env import MiniGridMultiStyles

# ---- config -----------------------------------------------------------------
CKPT_GLOB = os.path.join(os.path.dirname(__file__), "..", "trained_models", "*.pt")
ROUNDS_PER_CKPT = 5
MAX_STEPS = 100
STEP_DELAY = 0.08          # seconds between rendered steps (slow enough to watch)
SAMPLE_ACTIONS = True      # True = sample (as PPO did); False = greedy argmax
DEVICE = "cpu"


def build_actor(state_dict):
    """Rebuild the FCAgent actor MLP from a checkpoint state_dict and load it.

    Layer sizes are read straight from the weights, so this needs no env/config.
    """
    actor_sd = {
        k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")
    }
    num_obs = actor_sd["1.weight"].shape[1]      # Linear(num_obs, hidden)
    hidden = actor_sd["1.weight"].shape[0]
    num_actions = actor_sd["5.weight"].shape[0]  # Linear(hidden, num_actions)

    actor = nn.Sequential(
        nn.Flatten(),                 # 0
        nn.Linear(num_obs, hidden),   # 1
        nn.Tanh(),                    # 2
        nn.Linear(hidden, hidden),    # 3
        nn.Tanh(),                    # 4
        nn.Linear(hidden, num_actions),  # 5
    )
    actor.load_state_dict(actor_sd)
    actor.eval().to(DEVICE)
    return actor


def obs_to_tensor(obs):
    """Match ppo/util.preprocess_images: raw image as float32, add batch dim."""
    img = np.asarray(obs["image"], dtype=np.float32)   # (view, view, 3)
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)  # (1, view, view, 3)


def play_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    env_cfg = json.loads(ckpt["environment_config"])
    style = env_cfg.get("env_mode")
    corridor = env_cfg.get("bypass_corridor")
    view_size = env_cfg.get("view_size", 7)

    actor = build_actor(ckpt["model_state_dict"])

    name = os.path.basename(path)
    print(f"\n=== {name} | style={style} corridor={corridor} view={view_size} ===")

    env = MiniGridMultiStyles(
        target_style=style,
        target_bonus=1.0,
        non_target_penalty=-1.0,
        free_item_placement=True,
        bypass_corridor=corridor,
        agent_view_size=view_size,
        max_steps=MAX_STEPS,
        render_mode="human",
    )

    try:
        for ep in range(ROUNDS_PER_CKPT):
            obs, _ = env.reset(seed=1000 + ep)
            env.render()
            done = False
            ret = 0.0
            steps = 0
            info = {}
            while not done and steps < MAX_STEPS:
                with torch.no_grad():
                    logits = actor(obs_to_tensor(obs))
                    if SAMPLE_ACTIONS:
                        action = int(torch.distributions.Categorical(logits=logits).sample())
                    else:
                        action = int(torch.argmax(logits, dim=-1))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ret += float(reward)
                steps += 1
                env.render()
                time.sleep(STEP_DELAY)

            achieved = info.get("achieved_style")
            if achieved is None and isinstance(info.get("episode_summary"), dict):
                achieved = info["episode_summary"].get("achieved_style")
            success = "OK" if achieved == style else "x "
            print(
                f"  round {ep + 1}/{ROUNDS_PER_CKPT}: "
                f"return={ret:6.2f}  steps={steps:3d}  achieved={achieved}  [{success}]"
            )
    finally:
        env.close()


def main():
    paths = sorted(glob.glob(CKPT_GLOB))
    if not paths:
        print(f"No checkpoints found at {CKPT_GLOB}")
        return
    print(f"Found {len(paths)} checkpoint(s).")
    for path in paths:
        play_checkpoint(path)


if __name__ == "__main__":
    main()