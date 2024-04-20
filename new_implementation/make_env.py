import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, OneHotPartialObsWrapper, ViewSizeWrapper
import numpy as np
from new_implementation.configs import EnvironmentConfig
from envs.double_goal_minigrid import DoubleGoalEnv


def make_env(config: EnvironmentConfig, seed: int, idx: int, run_name: str, mode: int = 1):
    """Return a function that returns an environment after setting up boilerplate."""

    # only one of fully observed or flat one hot can be true.
    assert not (
        config.fully_observed and config.one_hot_obs
    ), "Can't have both fully_observed and flat_one_hot."

    def thunk():
        kwargs = {}
        if config.render_mode:
            kwargs["render_mode"] = config.render_mode
        if config.max_steps:
            kwargs["max_steps"] = config.max_steps

        # env = gym.make(config.env_id, **kwargs)
        env = DoubleGoalEnv(mode=mode, agent_start_pos=None, agent_pov=config.view_size, **kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.capture_video and idx == 0:
            env = RenderResizeWrapper(env, 256, 256)
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                # Video every 50 runs for env #1
                episode_trigger=lambda x: x % config.video_frequency == 0,
                disable_logger=True,
            )

        # hard code for now!
        if config.env_id.startswith("MiniGrid"):
            if config.fully_observed:
                env = FullyObsWrapper(env)
            if config.view_size != 7:
                env = ViewSizeWrapper(env, agent_view_size=config.view_size)
            if config.one_hot_obs:
                env = OneHotPartialObsWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.run_name = run_name
        return env

    return thunk


class RenderResizeWrapper(gym.Wrapper):
    def __init__(self, env: MiniGridEnv, render_width=256, render_height=256):
        super().__init__(env)
        self.render_width = render_width
        self.render_height = render_height

    def render(self):
        if self.env.render_mode == "rgb_array":
            img = self.env.render()
            img = np.array(img)

            # Resize image
            img = self._resize_image(
                img, self.render_width, self.render_height
            )

            return img

        else:
            return self.env.render()

    def _resize_image(self, image, width, height):
        from PIL import Image

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Resize image
        image = image.resize((width, height), resample=Image.BILINEAR)

        # Convert back to numpy array
        image = np.array(image)

        return image
