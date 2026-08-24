"""
Collect trajectory datasets by rolling out trained PPO checkpoints online.

For every checkpoint in `trained_models/*.pt` this script:
  * rebuilds the PPO agent + the exact vectorized env it was trained in
    (MiniGridMultiStyles, hard randomize-layout family), constructing the env
    directly in `make_multistyle_env_thunk` so the full config is explicit,
  * rolls the (stochastic) policy out online,
  * accumulates observations / actions / rewards / dones / truncated / infos
    (the per-step env `info` dicts carry the style summary fields:
    `episode_summary`, `achieved_style`, detection / lava / enemy counters, ...),
  * writes one gzip-pickled dataset per checkpoint, with the SAME on-disk schema
    produced during PPO data collection (see `TrajectoryWriter.write`).

Each dataset is guaranteed to hold at least `--min-samples` rollout rows
(first dim of `observations`); every row bundles `num_envs` parallel
transitions, so the actual transition count is `rows * num_envs`.

Output dir (default):
    datasets/minigrid/multi_style_env_hard_randomize_layout/from_checkpoints/

Run from the repo root:
    python ppo/collect_from_checkpoints.py
or with options:
    python ppo/collect_from_checkpoints.py --min-samples 5000 --num-envs 8
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

# The ppo package imports its siblings by bare module name (`from agent import ...`)
# while make_env imports `from envs.multi_style_minigrid_env import ...`, so BOTH the repo
# root and the ppo/ dir need to be on sys.path regardless of how we're launched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym  # noqa: E402
import torch as t  # noqa: E402

from minigrid.wrappers import (  # noqa: E402
    FullyObsWrapper,
    OneHotPartialObsWrapper,
    ViewSizeWrapper,
)

from agent import get_agent  # noqa: E402
from configs import EnvironmentConfig, OnlineTrainConfig, RunConfig  # noqa: E402
from envs.multi_style_minigrid_env import MiniGridMultiStyles  # noqa: E402
from memory import Memory  # noqa: E402
from trajectory_writer import TrajectoryWriter  # noqa: E402

# ---- config -----------------------------------------------------------------
CKPT_GLOB = os.path.join(_REPO_ROOT, "trained_models", "*.pt")
DEFAULT_OUT_DIR = os.path.join(
    _REPO_ROOT,
    "datasets",
    "minigrid",
    "multi_style_env_hard_randomize_layout",
    "from_checkpoints",
)
DEFAULT_MIN_SAMPLES = 25000      # minimum rollout rows per dataset
DEFAULT_NUM_ENVS = 8            # parallel envs (matches the PPO training runs)
SAMPLING_METHOD = "epsilon"     # policy Categorical sample, with EPSILON random noise
EPSILON = 0.01                  # chance of a uniform random action (extra diversity)


def out_path_for(checkpoint_path: str, out_dir: str) -> str:
    """`.../multi_style_env_hard_bypass_lower_02_PPO.pt`
    -> `<out_dir>/PPO_trajectories_multi_style_env_bypass_lower.gz`.

    Drops the `_PPO` suffix, the `_hard` qualifier, and the trailing version
    tag (e.g. `_02`)."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem.endswith("_PPO"):
        stem = stem[: -len("_PPO")]
    stem = stem.replace("_hard", "")        # drop the "_hard" qualifier
    stem = re.sub(r"_\d+$", "", stem)        # drop the trailing version tag (_02)
    return os.path.join(out_dir, f"PPO_trajectories_{stem}.gz")


def make_multistyle_env_thunk(environment_config, seed: int, idx: int, run_name: str):
    """Return a thunk that builds the exact env a checkpoint was trained in.

    This inlines what `ppo.make_env.make_env` did, but constructs
    `MiniGridMultiStyles` directly so every env setting is visible in one place.
    Video capture is intentionally omitted (collection never records video).
    """
    def thunk():
        # --- core env: all gameplay/reward settings explicit ----------------
        env = MiniGridMultiStyles(
            target_style=environment_config.env_mode,
            target_bonus=1.0,
            non_target_penalty=-1.0,
            free_item_placement=True,
            bypass_corridor=environment_config.bypass_corridor,
            agent_view_size=3,
            max_steps=130,
            render_mode=environment_config.render_mode,
        )

        # --- wrappers (match the training-time observation pipeline) ---------
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if environment_config.fully_observed:
            env = FullyObsWrapper(env)
        if environment_config.view_size != 7:
            env = ViewSizeWrapper(env, agent_view_size=environment_config.view_size)
        if environment_config.one_hot_obs:
            env = OneHotPartialObsWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.run_name = run_name
        return env

    return thunk


def build_agent(checkpoint_path: str, num_envs: int, device: str):
    """Rebuild the PPO agent + its vectorized env from a saved checkpoint.

    Mirrors `agent.load_saved_checkpoint`, but builds the env directly via
    `make_multistyle_env_thunk` (instead of `make_env`) so the full env config
    is visible here, and disables video capture so rollouts don't spew files.
    Returns (agent, environment_config, online_config).
    """
    saved_state = t.load(checkpoint_path, map_location=t.device("cpu"), weights_only=False)
    for key in ("environment_config", "model_config", "model_state_dict", "online_config"):
        assert key in saved_state, f"{key} not found in checkpoint"

    env_args = json.loads(saved_state["environment_config"])
    env_args["device"] = device
    env_args["capture_video"] = False
    environment_config = EnvironmentConfig(**env_args)

    online_args = json.loads(saved_state["online_config"])
    online_args.pop("batch_size", None)
    online_args.pop("minibatch_size", None)
    online_args["num_envs"] = num_envs
    online_args["device"] = device
    online_config = OnlineTrainConfig(**online_args)

    envs = gym.vector.AsyncVectorEnv(
        [
            make_multistyle_env_thunk(
                environment_config,
                seed=environment_config.seed,
                idx=i,
                run_name="from_checkpoints",
            )
            for i in range(num_envs)
        ],
        shared_memory=False,
    )

    agent = get_agent(
        envs=envs,
        environment_config=environment_config,
        online_config=online_config,
    )
    agent.load_state_dict(saved_state["model_state_dict"])
    agent.eval()
    return agent, environment_config, online_config


def collect_one(checkpoint_path: str, out_dir: str, min_samples: int, num_envs: int):
    trajectory_path = out_path_for(checkpoint_path, out_dir)
    print(f"\n=== {os.path.basename(checkpoint_path)} -> {trajectory_path} ===")

    device = "cuda" if t.cuda.is_available() else "cpu"
    agent, environment_config, online_config = build_agent(
        checkpoint_path, num_envs=num_envs, device=device
    )
    memory = Memory(agent.envs, online_config, device=agent.device)

    writer = TrajectoryWriter(
        path=trajectory_path,
        run_config=RunConfig(track=False),
        environment_config=agent.environment_config,
        online_config=online_config,
        model_config=agent.model_config,
    )

    # Each rollout step accumulates one row (num_envs transitions). Round the
    # requested minimum up to a whole number of steps.
    rollout_length = int(math.ceil(min_samples))
    print(f"Rolling out {rollout_length} steps x {num_envs} envs "
          f"(>= {min_samples} rows, ~{rollout_length * num_envs} transitions) ...")
    t0 = time.time()
    agent.rollout(
        memory=memory,
        num_steps=rollout_length,
        envs=agent.envs,
        trajectory_writer=writer,
        sampling_method=SAMPLING_METHOD,
        epsilon=EPSILON,
    )

    writer.tag_terminated_trajectories()
    writer.write(upload_to_wandb=False)
    agent.envs.close()
    rows = len(writer.observations)
    print(f"  done in {time.time() - t0:.1f}s | rows={rows} | "
          f"transitions={rows * num_envs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-glob", default=CKPT_GLOB,
        help="Glob for checkpoint .pt files (default: trained_models/*.pt).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Directory to write the .gz datasets into.",
    )
    parser.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
        help="Minimum number of rollout rows per dataset (default: 5000).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=DEFAULT_NUM_ENVS,
        help="Number of parallel envs per rollout (default: 8).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(args.checkpoints_glob))
    if not paths:
        print(f"No checkpoints found at {args.checkpoints_glob}")
        return
    print(f"Found {len(paths)} checkpoint(s). Writing datasets to {args.out_dir}")

    failures = []
    for path in paths:
        try:
            collect_one(path, args.out_dir, args.min_samples, args.num_envs)
        except Exception as exc:  # keep going across checkpoints
            print(f"  !! FAILED {os.path.basename(path)}: {exc!r}")
            failures.append((path, exc))

    print(f"\nDone. {len(paths) - len(failures)}/{len(paths)} datasets written.")
    if failures:
        for path, exc in failures:
            print(f"  failed: {os.path.basename(path)} -> {exc!r}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time() - start:.1f}s")
