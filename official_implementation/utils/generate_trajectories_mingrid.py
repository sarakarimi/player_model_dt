import sys
import random
import argparse
from typing import List

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from envs.double_goal_minigrid import DoubleGoalEnv
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper

from utils.minigrid_wrappers import FullyObsFeatureWrapper

sys.modules["gym"] = gym


def compute_reward_to_go(
    rewards: List[float], gamma: float = 0.999, discounted: bool = False
) -> List[float]:
    reward_to_go = []
    total_reward = 0

    for r in reversed(rewards):
        total_reward += r if not discounted else r + gamma * total_reward
        reward_to_go.insert(
            0, total_reward
        )  # Insert at the beginning to maintain order

    return reward_to_go


def generate_traj(
    model: BaseAlgorithm,
    save_path: str = None,
    n_episodes: int = 100,
    epsilon: float = 0.1,
    gamma: float = 0.999,
) -> None:
    observations = []
    timesteps = []
    actions = []
    rewards = []
    rewards_to_go = []
    next_observations = []
    dones = []
    ep_idx = 0

    episode_observations = []
    episode_timesteps = []
    episode_actions = []
    episode_rewards = []
    episode_next_observations = []
    episode_dones = []
    timestep = 0
    obs = model.env.reset()

    while ep_idx < n_episodes:
        episode_observations.append(obs[0])
        episode_timesteps.append(timestep)
        if random.random() > epsilon:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([model.env.action_space.sample()])

        next_obs, reward, done, _ = model.env.step(action)

        episode_actions.append(action[0])
        episode_rewards.append(reward[0])
        episode_dones.append(int(done[0]))
        episode_next_observations.append(next_obs[0])

        if done:
            obs = model.env.reset()

            # Compute reward-to-go
            reward_to_go = compute_reward_to_go(rewards=episode_rewards, gamma=gamma)
            observations.append(np.array(episode_observations))
            timesteps.append(np.array(episode_timesteps).reshape((-1, 1)))
            actions.append(np.array(episode_actions).reshape((-1, 1)))
            rewards.append(np.array(episode_rewards).reshape((-1, 1)))
            rewards_to_go.append(np.array(reward_to_go).reshape((-1, 1)))
            next_observations.append(np.array(episode_next_observations))
            dones.append(np.array(episode_dones).reshape((-1, 1)))

            episode_observations = []
            episode_next_observations = []
            episode_actions = []
            episode_rewards = []
            episode_timesteps = []
            episode_dones = []

            ep_idx += 1
            if ep_idx % 100 == 0:
                print(f"Episode: {ep_idx}/{n_episodes}")
            timestep = 0
        else:
            obs = next_obs
            timestep += 1

    timesteps = np.array(np.concatenate(timesteps, axis=0))
    observations = np.array(np.concatenate(observations, axis=0))
    actions = np.array(np.concatenate(actions, axis=0))
    rewards = np.array(np.concatenate(rewards, axis=0))
    rewards_to_go = np.array(np.concatenate(rewards_to_go, axis=0))
    next_observations = np.array(np.concatenate(next_observations, axis=0))
    dones = np.array(np.concatenate(dones, axis=0))

    # pytype: disable=attribute-error
    numpy_dict = {
        "timestep": timesteps,
        "obs": observations,
        "action": actions,
        "reward": rewards,
        "reward_to_go": rewards_to_go,
        "next_obs": next_observations,
        "dones": dones,
    }

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    model.env.close()
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generation script of trajectories of agents on the double goal minigrid environment."
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        help="Double goal minigrid env mode (0 - Double goals; 1 - Upper goal; 2 - Lower goal)",
        required=True,
    )
    parser.add_argument(
        "-n", "--num_episodes", type=int, help="Number of episodes", required=True
    )
    parser.add_argument(
        "-eps", "--epsilon", type=float, help="Epsilon (Action choice)", default=0.1
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        type=float,
        help="Gamma (Reward discount factor)",
        default=0.99,
    )
    parser.add_argument(
        "-dr",
        "--discount_reward",
        type=bool,
        help="Discount reward for reward-to-go?",
        default=False,
    )
    parser.add_argument(
        "-img",
        "--img",
        type=bool,
        help="Boolean flag to control observation type (True: Image observations, False: State features)",
        default=False,
    )
    parser.add_argument(
        "-num_steps",
        "--num_steps",
        type=int,
        help="Number of training steps",
        default=100000,
    )
    args = parser.parse_args()
    assert 0 <= args.mode < 3 and isinstance(args.mode, int)

    # Create environment
    if args.img:
        env = RGBImgObsWrapper(
            DoubleGoalEnv(
                mode=args.mode,
                render_mode="rgb_array",
                agent_start_pos=None,
                max_steps=50,
            )
        )
        env = ImgObsWrapper(env)
        model = PPO("CnnPolicy", env, verbose=1, n_epochs=50)
    else:
        env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=args.mode,
                render_mode="rgb_array",
                agent_start_pos=None,
                max_steps=50,
            )
        )
        model = PPO("MlpPolicy", env, verbose=1)

    # Load model weights
    model.set_parameters(
        f"trained_models/double_goal_mode_{args.mode}_image_ft_{int(args.img)}_nsteps_{args.num_steps}_ppo.zip",
        device="cpu",
    )

    # Generate trajectories
    generate_traj(
        model,
        f"data/double_goal_mode_{args.mode}_"
        f"  image_ft_{int(args.img)}_nsteps_{args.num_steps}_ppo_num_episodes{args.num_episodes}_eps_{args.epsilon}_gamma_{args.gamma}_dr_{args.discount_reward}",
        n_episodes=args.num_episodes,
        epsilon=args.epsilon,
        gamma=args.gamma,
    )
    env.close()


if __name__ == "__main__":
    main()
