import os
import sys
import argparse
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from envs.double_goal_minigrid import DoubleGoalEnv
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from utils.minigrid_wrappers import FullyObsFeatureWrapper

sys.modules["gym"] = gym


def rename_file(folder_path: str, old_name: str, new_name: str) -> None:
    # Construct the full paths for the old and new names
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    try:
        # Rename the file
        os.rename(old_path, new_path)
        print(f"File '{old_name}' renamed to '{new_name}' successfully.")
    except FileNotFoundError:
        print(f"File '{old_name}' not found in the folder.")
    except FileExistsError:
        print(f"A file with the name '{new_name}' already exists in the folder.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training script of agents on the double goal minigrid environment."
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        help="Double goal minigrid env mode (0 - Double goals; 1 - Upper goal; 2 - Lower goal)",
        required=True,
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
            ImgObsWrapper(
                DoubleGoalEnv(
                    mode=args.mode,
                    render_mode="rgb_array",
                    agent_start_pos=None,
                    max_steps=50,
                )
            )
        )
        test_env = RGBImgObsWrapper(
            ImgObsWrapper(
                DoubleGoalEnv(
                    mode=args.mode,
                    render_mode="rgb_array",
                    agent_start_pos=None,
                    max_steps=50,
                )
            )
        )
        model = PPO("CnnPolicy", env, verbose=1, n_epochs=50)
    else:
        env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=args.mode, agent_start_pos=None, max_steps=50, agent_pov=9
            )
        )
        test_env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=args.mode, agent_start_pos=None, max_steps=50, agent_pov=9
            )
        )
        model = PPO("MlpPolicy", env, verbose=1)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path="./trained_models/",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=100,
        deterministic=True,
        render=False,
    )

    # Agent
    model.learn(total_timesteps=args.train_steps, callback=eval_callback)

    # Change name of saved model
    rename_file(
        folder_path="./trained_models/",
        old_name="best_model.zip",
        new_name=f"double_goal_mode_{args.mode}_image_ft_{int(args.img)}_nsteps_{args.train_steps}_ppo.zip",
    )

    test_env.close()
    env.close()


if __name__ == "__main__":
    main()
