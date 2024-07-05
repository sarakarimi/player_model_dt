
from __future__ import annotations

from envs.minigrid_dungeon.core.grid import Grid
from envs.minigrid_dungeon.core.mission import MissionSpace
from envs.minigrid_dungeon.core.world_object import Goal, Treasure, Weapon, Enemy, Lava
from envs.minigrid_dungeon.manual_control import ManualControl
from envs.minigrid_dungeon.minigrid_env import MiniGridDungeonEnv


class FourRoomsTreasureEnemyEnv(MiniGridDungeonEnv):
    """
    ## Description

    This environment is sequence of four connected rooms, and the goal of the agent is to reach the
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

    def __init__(self,
                agent_pos=None,
                goal_pos=None,
                max_steps=100,
                reward_treasure: bool = True,
                reward_enemy: bool = True,
                **kwargs):
        
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.size = 12
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            see_through_walls=True,
            agent_view_size=11,
            reward_treasure=reward_treasure,
            reward_enemy=reward_enemy,
            attack_chance=1.0,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal, pick up treasure and kill the enemy."

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        # Place a treasure square in the bottom-left corner
        self.place_obj(Treasure())
        self.num_treasures = 1

        # Place a weapon square in the top-right corner
        self.place_obj(Weapon())

        # Place a enemy square in the middle
        self.place_obj(Enemy())
        self.num_enemies = 1

        # Place lava randomly
        self.place_obj(Lava())
        self.place_obj(Lava())


def main():
    env = FourRoomsTreasureEnemyEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
