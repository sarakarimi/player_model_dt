from __future__ import annotations
from typing import Optional, Tuple

import gymnasium as gym
import pygame
from gymnasium import spaces
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Goal, Wall, Ball, Box
from minigrid.core.grid import Grid
from minigrid.core.constants import DIR_TO_VEC

from envs.metalgridsolid.utils.utils import draw_triangle
from envs.old.minigrid_dungeon.manual_control import ManualControl


# --- Custom objects -----------------------------------------------------------

class Enemy(WorldObj):
    """
    Simple enemy with a facing direction and a short vision cone.
    The enemy doesn't move in this minimal example (you can extend later).
    """

    def __init__(self, color="blue", dir=3):
        super().__init__("box", color)
        self.dir = dir  # 0:right, 1:down, 2:left, 3:up

    def can_overlap(self):
        return False

    # def see_behind_vec(self):
    #     # Vector pointing "behind" relative to enemy facing
    #     # dir^2 toggles 180 degrees: (dir + 2) % 4
    #     return DIR_TO_VEC[(self.dir + 2) % 4]

    def render(self, img):
        # Draw a filled triangle pointing in self.dir

        color = (256, 256, 200)
        tri_dir = {0: 1, 1: 2, 2: 3, 3: 0}[self.dir]

        cell_size = img.shape[0]  # tile is square
        draw_triangle(
            position=(0, 0),  # tile-local
            direction=tri_dir,
            color=color,
            img=img,
            cell_size=cell_size,
            scale=0.8,  # a touch bigger looks nicer
        )


class Weapon(WorldObj):
    """
    Simple weapon; agent can carry it. We use Ball semantics for pickup,
    but give it a distinct type/name.
    """

    def __init__(self, color="yellow"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        # Draw like a yellow circle-ish
        img[:, :, 0] = 255
        img[:, :, 1] = 255
        img[:, :, 2] = 0


# 1. Add Camouflage object
class Camouflage(WorldObj):
    """
    Camouflage object; agent can pick it up to hide from the enemy when passing the enemy exposure zone.
    """

    def __init__(self, color="red"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        img[:, :, 0] = 255
        img[:, :, 1] = 141
        img[:, :, 2] = 161


# --- Environment --------------------------------------------------------------

class MiniGridThreeStyles(MiniGridEnv):
    """
    Three valid styles to reach the same goal:
      - bypass: reach goal without ever entering the enemy's vision cone
      - backstab: stand directly behind enemy and toggle to neutralize
      - weapon: pick up weapon and toggle adjacent to enemy to neutralize

    You succeed by reaching the goal. The env reports which style you used
    in info["style"] = {"bypass" | "backstab" | "weapon"}.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
            self,
            size: int = 7,
            max_steps: Optional[int] = None, *,
            target_style: Optional[str] = None,  # "bypass" | "backstab" | "weapon" | None
            target_bonus: float = 0.6,  # paid if achieved == target_style
            non_target_penalty: float = 0.0,  # paid if achieved != target_style
            style_bonuses: Optional[dict] = None,  # used only when target_style is None
            easy_env: bool = True,  # easy 3 style env with no pillars and simple layout
            **kwargs
    ):

        # TODO remove
        self.weapon_picked = False
        self.weapon_dropped = False
        self.object_pickup_bonus = 0.0
        self.killing_bonus = 0.0
        self.in_exposure_with_camouflage = False
        self.camouflage_picked = False
        self.camouflage_dropped = False

        self.size = size
        self.easy_env = easy_env
        if not self.easy_env:
            self.size = 11
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(mission_space=mission_space, grid_size=self.size, max_steps=max_steps or 4 * size * size,
                         **kwargs)
        # State flags
        self.detected = False
        self.enemy_pos: Optional[Tuple[int, int]] = None
        self.enemy_dir: int = 0
        self.enemy_alive = True
        self.enemy_obj: Optional[Enemy] = None
        self.style_used: Optional[str] = None

        # NEW: end immediately on detection (and optional penalty)
        self.end_on_detection: bool = True
        self.detection_penalty: float = 0.0 #-0.2 #-1.0  # set to e.g. -1.0 if you want a penalty

        # --- simple style config (NO modes) ---
        if self.easy_env:
            assert target_style in {None, "bypass", "backstab", "weapon"}
        else:
            assert target_style in {None, "bypass", "weapon", "camouflage"}
        self.target_style: Optional[str] = target_style
        self.target_bonus: float = target_bonus
        self.non_target_penalty: float = non_target_penalty

        # defaults when no target_style is specified
        self.weapon_bonus = 0.6
        self.camouflage_bonus = 0.6  # Similar to weapon
        self.backstab_bonus = 0.6
        self.bypass_bonus = 0.6

        # Set valid styles and bonuses per mode
        if self.easy_env:
            assert self.target_style in {None, "bypass", "backstab", "weapon"}
            defaults = {"bypass": self.bypass_bonus, "backstab": self.backstab_bonus, "weapon": self.weapon_bonus}
        else:
            assert self.target_style in {None, "bypass", "weapon", "camouflage"}
            defaults = {"bypass": self.bypass_bonus, "weapon": self.weapon_bonus, "camouflage": self.camouflage_bonus}
        if style_bonuses:
            defaults.update(style_bonuses)
        self.style_bonuses = defaults


    @staticmethod
    def _gen_mission() -> str:
        return "three_style_policy"

    def _create_grid(self, width, height) -> Grid:
        return Grid(width, height)

    def _gen_grid(self, width, height):
        # TODO remove
        self.weapon_picked = False
        self.weapon_dropped = False
        self.object_pickup_bonus = 0.0
        self.killing_bonus = 0.0
        self.in_exposure_with_camouflage = False
        self.camouflage_picked = False
        self.camouflage_dropped = False

        self.grid = self._create_grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Layout:
        # Agent starts on the left, goal on the far right.
        # Enemy approximately in the middle, facing right.
        # Weapon somewhere else (top-left-ish).
        # Keep things open so bypass vs backstab vs weapon are all feasible.

        # Place goal
        goal_x, goal_y = width - 2, height // 2
        self.goal_pos = (goal_x, goal_y)

        self.put_obj(Goal(), goal_x, goal_y)

        # Place enemy
        ex, ey = width // 2, height // 2
        self.enemy_dir = 0  # facing right
        self.enemy_obj = Enemy(dir=self.enemy_dir)
        self.put_obj(self.enemy_obj, ex, ey)
        self.enemy_pos = (ex, ey)
        self.enemy_alive = True

        # Place weapon
        if self.easy_env:
            wx, wy = 2, 1
        else:
            wx, wy = 3, 5
        self.put_obj(Weapon(), wx, wy)

        # Place Camouflage
        if not self.easy_env:
            self.put_obj(Camouflage(), 1, 3)  # Add this line

        # Optionally add some walls to make routing interesting but not forced
        # (Keep simple; you can tweak as needed)
        # Example: a little pillar above the enemy to shape paths
        if not self.easy_env:
            px, py = ex, ey + 1
            if 1 < py < height - 1:
                for i in range(4):
                    self.put_obj(Wall(), px + i, py)
                    self.put_obj(Wall(), px + i, py + 2)
                self.put_obj(Wall(), px, py + 1)

        # Place agent
        self.place_agent(top=(1, height // 2 + 2), size=(1, 1))

        self.detected = False
        self.style_used = None
        self.mission = (
            "Reach the goal via one of three styles: "
            "(1) bypass unseen, (2) backstab from behind with toggle, "
            "(3) pick up weapon and toggle to defeat enemy."
        )


    # --- Helpers --------------------------------------------------------------

    def _achieved_style(self) -> str:
        # In non-easy env, camouflage is a valid style
        if not self.easy_env:
            if self.style_used is not None:
                return self.style_used
            if self.enemy_alive and self._agent_has_camouflage() and self.agent_pos == self.goal_pos and self.in_exposure_with_camouflage:
                return "camouflage"
            # fallback
            return "bypass"
        else:
            # If enemy is alive and we never set style_used, treat as bypass
            return self.style_used if self.style_used is not None else ("bypass" if self.enemy_alive else "bypass")

    def _enemy_dir(self):
        # Use the Enemy object's dir when available
        if self.enemy_obj is not None:
            return self.enemy_obj.dir
        return self.enemy_dir

    def _front_tile(self):
        """Tile directly in front of the enemy, based on its current dir."""
        if not self.enemy_pos:
            return None
        ex, ey = self.enemy_pos
        dx, dy = DIR_TO_VEC[self._enemy_dir()]
        return (ex + dx, ey + dy)

    def _is_in_enemy_front_tile(self, agent_pos):
        """True iff agent stands on the single tile in front of the enemy."""
        ft = self._front_tile()
        return ft is not None and tuple(agent_pos) == ft

    def is_in_custom_enemy_box(self, agent_pos: Tuple[int, int]) -> bool:
        if not self.enemy_alive or self.enemy_pos is None:
            return False

        ex, ey = self.enemy_pos
        ax, ay = agent_pos

        # Rectangle: from (ex, ey) to (ex+3, ey-5)
        in_x = ex <= ax <= ex + 4
        in_y = ey - 5 <= ay <= ey

        return in_x and in_y and (ax, ay) != self.goal_pos

    def _is_adjacent(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _is_behind_enemy(self, agent_pos: Tuple[int, int]) -> bool:
        """Agent tile matches the tile directly behind the enemy."""
        if self.enemy_pos is None:
            return False
        ex, ey = self.enemy_pos
        bx, by = DIR_TO_VEC[(self.enemy_dir + 2) % 4]  # behind vector
        return agent_pos == (ex + bx, ey + by)

    def _agent_has_weapon(self) -> bool:
        return self.carrying is not None and isinstance(self.carrying, Weapon)

    def _remove_enemy(self, method: str):
        """Remove enemy from grid and record the style method."""
        if self.enemy_pos:
            x, y = self.enemy_pos
            self.grid.set(x, y, None)
        self.enemy_alive = False
        self.style_used = method

    def _agent_has_camouflage(self) -> bool:
        return self.carrying is not None and isinstance(self.carrying, Camouflage)

    # --- Step override --------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward -= 0.005  # step penalty: comment out for Bypass mode or camouflage longer paths

        # TODO remove Weapon pickup/drop tracking for rewards/penalties
        has_weapon = self._agent_has_weapon()
        # pickup reward (first time only)
        if (not self.weapon_picked) and has_weapon:
            reward += 0.2 if self.target_style == "weapon" else 0.0
            self.weapon_picked = True
            self.weapon_dropped = False

        # drop penalty (first drop after pickup, only once)
        if self.weapon_picked and (not has_weapon) and (not self.weapon_dropped):
            reward -= 0.2 if self.target_style == "weapon" else 0.0
            self.weapon_dropped = True


        # TODO remove Camouflage pickup/drop tracking for rewards/penalties
        has_camouflage = self._agent_has_camouflage()
        if (not self.camouflage_picked) and has_camouflage:

            reward += 0.2 if self.target_style == "camouflage" else 0.0
            self.camouflage_picked = True
            self.camouflage_dropped = False
        if self.camouflage_picked and (not has_camouflage) and (not self.camouflage_dropped):
            reward -= 0.2 if self.target_style == "camouflage" else 0.0
            self.camouflage_dropped = True
        # Update exposure flag
        if has_camouflage and self.is_in_custom_enemy_box(self.agent_pos):
            self.in_exposure_with_camouflage = True
        else:
            if self.agent_pos != self.goal_pos:
                self.in_exposure_with_camouflage = False


        if self.easy_env:
            enemy_exposure_function = self._is_in_enemy_front_tile
            # Track detection if enemy alive
            if self.enemy_alive and enemy_exposure_function(self.agent_pos):
                self.detected = True  # just a flag; doesn't end the episode by itself
                if self.end_on_detection:
                    terminated = True
                    reward = self.detection_penalty
                    info = dict(info)
                    info["termination"] = "detected_front_tile"
                    info["style"] = None
                    info["detected"] = True
                    info["enemy_alive"] = self.enemy_alive
                    return obs, reward, terminated, truncated, info

            # Handle toggling near enemy for backstab/weapon
            if action == self.actions.toggle and self.enemy_alive and self.enemy_pos:
                if self._is_adjacent(self.agent_pos, self.enemy_pos):
                    if self._is_behind_enemy(self.agent_pos):
                        # backstab
                        self._remove_enemy("backstab")
                    if self._agent_has_weapon():
                        # weapon attack from any adjacent tile
                        self._remove_enemy("weapon")

        else:
            enemy_exposure_function = self.is_in_custom_enemy_box
            # Track detection if enemy alive
            if self.enemy_alive and enemy_exposure_function(self.agent_pos) and not self._agent_has_camouflage():
                self.detected = True  # just a flag; doesn't end the episode by itself
                if self.end_on_detection:
                    terminated = True
                    reward = self.detection_penalty
                    info = dict(info)
                    info["termination"] = "detected_front_tile"
                    info["style"] = None
                    info["detected"] = True
                    info["enemy_alive"] = self.enemy_alive
                    return obs, reward, terminated, truncated, info

            # Handle toggling near enemy for weapon
            if action == self.actions.toggle and self.enemy_alive and self.enemy_pos:
                if self._is_adjacent(self.agent_pos, self.enemy_pos):
                    if self._agent_has_weapon():
                        # weapon attack from any adjacent tile
                        self._remove_enemy("weapon")
                        if self.target_style == "weapon":
                            reward += 0.2

        # If we reached the goal, decide style if not already set
        info = dict(info)
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Goal):
            terminated = True
            achieved = self._achieved_style()
            if self.target_style is None:
                # No target: pay per-style default bonus
                bonus = self.style_bonuses.get(achieved, 0.0)
            else:
                # Targeted training: pay one number for match, another for mismatch
                bonus = self.target_bonus if achieved == self.target_style else self.non_target_penalty

            base = self._reward()
            reward = base + bonus

            info["target_style"] = self.target_style
            info["achieved_style"] = achieved
            info["base_reward"] = base
            info["style_bonus_or_penalty"] = bonus
            info["total_reward"] = reward
            return obs, reward, terminated, truncated, info


        info["style"] = self.style_used
        info["detected"] = self.detected
        info["enemy_alive"] = self.enemy_alive
        return obs, reward, terminated, truncated, info


# --- Registration helper ------------------------------------------------------

def register_env():
    gym.envs.registration.register(
        id="MiniGrid-ThreeStyles-v0",
        entry_point=MiniGridThreeStyles,
    )


if __name__ == '__main__':
    register_env()
    import gymnasium as gym

    env = gym.make("MiniGrid-ThreeStyles-v0", target_style="bypass", target_bonus=1.0, non_target_penalty=-1.0,
                   render_mode="human", easy_env=False, max_steps=100)
    obs, _ = env.reset()
    ret = 0.0
    finish = False
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0  # Rotate Left
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Rotate Right
                elif event.key == pygame.K_UP:
                    action = 2  # Move Forward
                elif event.key == pygame.K_SPACE:
                    action = 3  # Pick Up Item
                elif event.key == pygame.K_TAB:
                    action = 4  # Drop an Item
                elif event.key == pygame.K_DOWN:
                    action = 5  # Use Weapon
                elif event.key == pygame.K_z:
                    action = 6  # Done
                else:
                    action = None

                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
                    ret += reward
                    print(info)
                    print(reward)
                    finish = done or truncated
                print(ret)

        env.render()
        env.clock.tick(10)
