import numpy as np
import gymnasium as gym
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

OBJECT_TO_IDX = {
    "empty": 0,
    "wall": 1,
    "goal": 2,
    "lava": 3,
    "agent": 4,
}
class SymbolicPartialObsWrapper(ObservationWrapper):

    def __init__(self, env, agent_view_size):
        super().__init__(env)
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(self.env.width, self.env.height), dtype="uint8"
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):

        # env = self.unwrapped
        # grid = env.gen_obs_grid()
        # # grid, vis_mask = env.gen_obs_grid(self.agent_view_size)
        #
        # # Encode the partially observable view into a numpy array
        # # print(grid.grid)
        #
        # objects = np.array(
        #     [OBJECT_TO_IDX[o.type] if o is not None else 0 for o in grid.grid]
        # )
        # print(objects)
        # # grid = grid.encode(vis_mask)
        #
        # agent_pos = self.env.agent_pos
        # # ncol, nrow = self.width, self.height
        # # grid = np.mgrid[:ncol, :nrow]
        # # _objects = np.transpose(objects, (1, 0))
        #
        # grid = objects
        # # grid = np.transpose(grid, (1, 2, 0))
        # grid[agent_pos[0]][agent_pos[1]] = OBJECT_TO_IDX["agent"]
        # obs["image"] = grid

        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid = full_grid[:][:][0]
        print(full_grid.shape)
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = OBJECT_TO_IDX["agent"]


        return full_grid