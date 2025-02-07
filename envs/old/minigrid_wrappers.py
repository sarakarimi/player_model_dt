import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.vector.utils import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


class FullyObsFeatureWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.highlight = False

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype="uint8",
        )

        new_obs_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )
        self.observation_space = new_obs_space.spaces["image"]

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return full_grid
