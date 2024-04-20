import sys
import imageio
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from envs.double_goal_minigrid import DoubleGoalEnv
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from envs.minigrid_wrappers import FullyObsFeatureWrapper

sys.modules["gym"] = gym


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation script of agents on the double goal minigrid environment."
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        help="Double goal minigrid env mode (0 - Double goals; 1 - Upper goal; 2 - Lower goal)",
        required=True,
    )
    parser.add_argument(
        "-eval",
        "--eval_episodes",
        type=int,
        help="Number of evaluation episodes",
        default=100,
    )
    parser.add_argument(
        "-img",
        "--img",
        type=bool,
        help="Boolean flag to control observation type (True: Image observations, False: State features)",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--train_steps",
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
        f"trained_models/double_goal_mode_{args.mode}_image_ft_{int(args.img)}_nsteps_{args.train_steps}_ppo.zip",
        device="cuda",
    )
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        model.env,
        n_eval_episodes=args.eval_episodes,
        render=True,
        deterministic=False,
        return_episode_rewards=True,
        warn=False,
        callback=None,
    )

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
    print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

    # Get GIF of agent performance
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")
    blank_screen = np.zeros(shape=img.shape, dtype=np.uint8)
    images.append(np.asarray(blank_screen))
    for i in range(1000):
        images.append(img)
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = model.env.step(action)
        if done:
            images.append(np.asarray(blank_screen))
        img = model.env.render(mode="rgb_array")
    imageio.mimsave(
        f"trained_models/double_goal_mode_{args.mode}_image_ft_{int(args.img)}_nsteps_{args.train_steps}_ppo.gif",
        [np.array(img) for i, img in enumerate(images)],
        fps=4,
    )

    env.close()


if __name__ == "__main__":
    main()
