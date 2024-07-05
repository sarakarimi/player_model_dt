from __future__ import annotations

from envs.minigrid_dungeon.core.grid import Grid
from envs.minigrid_dungeon.core.mission import MissionSpace
from envs.minigrid_dungeon.core.world_object import Goal, Treasure, Weapon, Enemy, Lava
from envs.minigrid_dungeon.manual_control import ManualControl
from envs.minigrid_dungeon.minigrid_env import MiniGridDungeonEnv


class EmptyTreasureEnemyEnv(MiniGridDungeonEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, pick up the treasure and kill the enemy, which provides a sparse reward. 
    A small penalty is subtracted for the number of steps to reach the goal.

    ## Mission Space

    "get to the green goal square, pick up the treasure and kill the enemy."

    ## Action Space

    | Num | Name         | Action                      |
    |-----|--------------|-----------------------------|
    | 0   | left         | Turn left                   |
    | 1   | right        | Turn right                  |
    | 2   | forward      | Move forward                |
    | 3   | pickup       | Pickup Treasure and Weapons |
    | 4   | drop         | Unused                      |
    | 5   | toggle       | Unused                      |
    | 6   | done         | Unused                      |
    | 7   | attack       | Atack agent (in front)      |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    The reward is multi-component:
      - Goal Reward = '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
      - Treasure Reward = '1' for picking up the treasure, and '0' for failure.
      - Enemy Reward = '1' for killing the enemy, and '0' for failure.
    The total reward is computed as
    `reward = goal_reward * treasure_reward * enemy_reward'
    The reward is only given if the agent reaches the goal.
    Note: if reward_treasure or reward_enemy is set to False, the corresponding reward is not given (i.e., equal to one automatically).

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent is killed by the enemy.
    3. The agent falls into the lava.
    4. Timeout (see `max_steps`).
    """

    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        reward_treasure: bool = True,
        reward_enemy: bool = True,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=11,
            reward_treasure=reward_treasure,
            reward_enemy=reward_enemy,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal, pick up treasure and kill enemy."

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place a treasure square in the bottom-left corner
        self.put_obj(Treasure(), 1, height - 2)
        self.num_treasures = 1

        # Place a weapon square in the top-right corner
        self.put_obj(Weapon(), width - 2, 1)

        # Place a enemy square in the middle
        self.put_obj(Enemy(), int(width/2.), int(height/2.))
        self.num_enemies = 1

        # Place lava randomly
        self.place_obj(Lava())
        self.place_obj(Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square and pick up the treasure"

    
def main():
    env = EmptyTreasureEnemyEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()