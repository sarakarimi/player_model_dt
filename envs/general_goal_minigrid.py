from __future__ import annotations

import random
from enum import IntEnum
from typing import Tuple, Union, SupportsFloat, Any
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    up = 2
    # Pick up an object
    down = 3


class DoubleGoalEnv(MiniGridEnv):
    def __init__(
        self,
        size: int = 10,
        mode: int = 1,  # 0 - Double goals, 1 - Up goal, 2 - Down goal
        agent_start_pos: Union[Tuple[int, int], None] = None,
        agent_start_dir: int = 0,
        max_steps: Union[int, None] = None,
        **kwargs,
    ) -> None:
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            agent_view_size=9,
            max_steps=max_steps,
            **kwargs,
        )

        self.mode = mode
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission() -> str:
        return "bimodal_policy"

    def _gen_grid(self, width, height) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place two goals square (bottom-right corner and top-right corner)
        if self.mode == 0:
            self.put_obj(Goal(), width - 2, 1)
            self.put_obj(Goal(), width - 2, height - 2)
        elif self.mode == 1:
            self.put_obj(Goal(), width - 2, 1)
            self.put_obj(Lava(), width - 2, height - 2)
        elif self.mode == 2:
            self.put_obj(Lava(), width - 2, 1)
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0
        else:
            self.agent_pos = (1, random.randint(1, height-2))
            self.agent_dir = 0

        self.mission = "bimodal_policy"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False
        self.agent_dir = 0

        # Move up
        if action == self.actions.up:
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == self.actions.down:
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == self.actions.left:
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == self.actions.right:
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        else:
            raise ValueError(f"Unknown action: {action}")

        # Get the position in front of the agent
        fwd_pos = new_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = self._reward()
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True
            reward = -1

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


def main() -> None:
    env = DoubleGoalEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
