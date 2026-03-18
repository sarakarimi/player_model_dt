from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Union, SupportsFloat, Any, List
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.wrappers import SymbolicObsWrapper

# from envs.old.minigrid_wrappers import SymbolicPartialObsWrapper

"""

This module defines the MultiGoal environment for a grid-based game where an agent must navigate to specific goal positions. The environment is initialized with a specified number of goals, and the agent is tasked with targeting specific goal IDs.

Goal Positions:
The goals are positioned at specific locations within the grid, defined as follows:
1. Top-left corner
2. Bottom-right corner
3. Top-right corner
4. Bottom-left corner
5. Middle-top wall
7. Middle-bottom wall
6. Middle-right wall
8. Middle-left wall

The boolean eval_mode determines whether the non-active goals are replaced with lava or remain as goals. In eval_mode, all goals remain as goals!
"""


class MultiGoalManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        s, reward, terminated, truncated, _ = self.env.step(action)
        print(s['image'].shape, s['image'].tolist())
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": 0,
            "right": 1,
            "up": 2,
            "down": 3,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


class SimpleActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    up = 2
    down = 3


class MultiGoalEnv(MiniGridEnv):
    def __init__(
        self,
        size: int = 7,
        num_goals: int = 1,
        select_id_goal: Union[int, List[int]] = 0,
        agent_start_pos: Union[Tuple[int, int], None] = None,
        agent_start_dir: int = 0,
        max_steps: Union[int, None] = None,
        eval_mode: bool = False,
        **kwargs,
    ) -> None:
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            agent_view_size=3,
            max_steps=max_steps,
            **kwargs,
        )

        self.num_goals = num_goals
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.eval_mode = eval_mode

        if isinstance(select_id_goal, int):
            self.select_id_goal = [select_id_goal]
        else:
            self.select_id_goal = select_id_goal

        assert 0 < self.num_goals <= 8, "Maximum eight goals are supported"
        for goal_id in self.select_id_goal:
            assert 0 <= goal_id < self.num_goals, "Invalid goal id"

        self.actions = SimpleActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission() -> str:
        return "multi_modal_policy"

    def _gen_grid(self, width, height) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Define goal positions in corners and walls (clockwise from top-left corner)
        goal_positions = [
            (1, 1),              # Top-left corner
            (width - 2, height - 2),  # Bottom-right corner
            (width - 2, 1),       # Top-right corner
            (1, height - 2),      # Bottom-left corner
            (width // 2, 1),      # Middle-top wall
            (width - 2, height // 2), # Middle-right wall
            (width // 2, height - 2), # Middle-bottom wall
            (1, height // 2)      # Middle-left wall
        ]

         # Place goals based on num_goals
        selected_goal_positions = [goal_positions[i] for i in range(self.num_goals)]

        # Place the active goal and replace the rest with lava if not in eval_mode
        for i, pos in enumerate(selected_goal_positions):
            if i in self.select_id_goal:  # Now check if the goal ID is in the list
                self.put_obj(Goal(), *pos)  # Active goal
            else:
                if self.eval_mode:
                    self.put_obj(Goal(), *pos)  # In eval mode, all goals remain as goals
                else:
                    self.put_obj(Lava(), *pos)  # Non-active goals replaced with lava

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0
        else:
            self.agent_start_pos = (width // 2, height // 2)
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0

        self.mission = "multimodal_policy"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False
        # self.agent_dir = 0

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


    env = MultiGoalEnv(render_mode="human",
                       num_goals = 8,       # Maximum number of goals is 8
                       select_id_goal = [0, 1, 2, 3])


    # enable manual control for testing
    manual_control = MultiGoalManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
