from __future__ import annotations

from typing import Optional, Tuple
import random

import gymnasium as gym
import numpy as np
import pygame
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv


# --- rendering utils -----------------------------------------------------------
def fill_coords(img, fn, color):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return xmin <= x <= xmax and ymin <= y <= ymax
    return fn


# --- Custom world objects ------------------------------------------------------
# NB: the type strings below are reused only to obtain distinct object indices in
# obs channel 0 (MiniGrid's OBJECT_TO_IDX is fixed); behaviour comes from the
# methods and from the env logic, not the type string.

class Enemy(WorldObj):
    def __init__(self, color="blue", dir=0):
        super().__init__("box", color)
        self.dir = dir

    def can_overlap(self):
        return False

    def render(self, img):
        c = np.array([155, 89, 182])
        cx, cy, radius = 0.5, 0.5, 0.2
        num_spikes, spike_length, spike_width = 10, 0.3, 0.1

        def spiky_ball_fn(x, y):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                return True
            for i in range(num_spikes):
                angle = 2 * np.pi * i / num_spikes
                bx, by = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
                tx, ty = cx + (radius + spike_length) * np.cos(angle), cy + (radius + spike_length) * np.sin(angle)
                dx, dy = tx - bx, ty - by
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length == 0:
                    continue
                dx, dy = dx / length, dy / length
                p = (x - bx) * dx + (y - by) * dy
                if 0 <= p <= length and abs((x - bx) * dy - (y - by) * dx) <= spike_width / 2:
                    return True
            return False

        fill_coords(img, spiky_ball_fn, c)


class Weapon(WorldObj):
    def __init__(self, color="yellow"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = np.array([241, 196, 15])
        cx, cy, blade_h, handle_h, blade_w, handle_w, pommel_r = 0.5, 0.8, 0.7, 0.1, 0.1, 0.3, 0.07

        def sword_fn(x, y):
            if cy - blade_h <= y <= cy and cx - blade_w / 2 <= x <= cx + blade_w / 2:
                return True
            if cy - handle_h <= y <= cy and cx - handle_w / 2 <= x <= cx + handle_w / 2:
                return True
            return (x - cx) ** 2 + (y - (cy + pommel_r)) ** 2 <= pommel_r ** 2

        fill_coords(img, sword_fn, c)


class Camouflage(WorldObj):
    def __init__(self, color="red"):
        super().__init__("door", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.35 ** 2, (34, 85, 34))


class Boots(WorldObj):
    """While carried, the agent can traverse lava tiles safely."""

    def __init__(self, color="purple"):
        super().__init__("floor", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = np.array([90, 60, 30])
        fill_coords(img, point_in_rect(0.38, 0.55, 0.18, 0.7), c)
        fill_coords(img, point_in_rect(0.30, 0.80, 0.70, 0.85), c)
        fill_coords(img, point_in_rect(0.30, 0.80, 0.83, 0.88), (40, 25, 10))


class HazardTile(WorldObj):
    """Lava-like hazard (type 'ball' so MiniGrid's built-in lava death does not
    fire); death-unless-boots is handled by the env."""

    def __init__(self):
        super().__init__("ball", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), (255, 100, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.25, 0.35), (200, 60, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.55, 0.65), (200, 60, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.85, 0.95), (200, 60, 0))


class Portal(WorldObj):
    """Teleport tile. Uses type 'lava' purely for a distinct obs index; the env
    overrides the resulting built-in termination and teleports the agent from
    the entrance head to the exit head instead."""

    def __init__(self, color="blue"):
        super().__init__("lava", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, lambda x, y: 0.20 ** 2 <= (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.42 ** 2, (80, 120, 255))
        fill_coords(img, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.10 ** 2, (200, 220, 255))


# --- Environment ---------------------------------------------------------------

class MiniGridFourStyles(MiniGridEnv):
    """
    13x13 four-style MiniGrid env (extends the three-style env with a lava
    daredevil route and a one-way portal route). The goal is unreachable
    without one of the four mechanics, so there is no plain "bypass" route.

      weapon     - pick up the weapon, kill the enemy (toggle adjacent), reach goal.
      camouflage - pick up camo, pass through the detection zone, reach goal.
      daredevil  - pick up boots, traverse the lava, reach goal.
      portal     - step on portal head 1 -> teleport to head 2 (by the goal).

    Achievement priority: weapon > camouflage > daredevil > portal.

           x: 0  1  2  3  4  5  6  7  8  9 10 11 12
     y= 0:    #  #  #  #  #  #  #  #  #  #  #  #  #
     y= 1:    #  .  .  .  .  .  d  d  d  d  d  .  #
     y= 2:    #  .  .  .  .  .  d  d  d  d  d  .  #
     y= 3:    #  C  .  .  .  .  d  d  d  d  d  2  #
     y= 4:    #  .  .  .  .  .  d  d  d  d  d  .  #
     y= 5:    #  .  .  W  .  .  d  d  d  d  d  .  #
     y= 6:    #  .  .  B  .  .  E  d  d  d  d  G  #
     y= 7:    #  .  .  .  .  .  #  #  #  d  d  .  #
     y= 8:    #  A  .  .  .  .  #  #  #  .  .  .  #
     y= 9:    #  .  .  .  .  .  ~  ~  ~  ~  ~  .  #
     y=10:    #  .  .  .  .  .  ~  ~  ~  ~  ~  .  #
     y=11:    #  .  .  1  .  .  ~  ~  ~  ~  ~  .  #
     y=12:    #  #  #  #  #  #  #  #  #  #  #  #  #

    Legend: W weapon, C camo, B boots, E enemy, d detection, ~ lava,
    1 portal-in, 2 portal-out, A agent start, G goal.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    VALID_STYLES = ("weapon", "camouflage", "daredevil", "portal")

    def __init__(
        self,
        size: int = 13,
        max_steps: Optional[int] = None,
        *,
        target_style: Optional[str] = None,
        target_bonus: float = 0.6,
        non_target_penalty: float = 0.0,
        style_bonuses: Optional[dict] = None,
        randomize_layout: bool = False,
        end_on_detection: bool = True,
        end_on_lava: bool = True,
        detection_penalty: float = 0.0,
        lava_penalty: float = 0.0,
        pickup_bonus: float = 0.2,
        drop_penalty: float = -0.2,
        kill_bonus: float = 0.2,
        first_lava_bonus: float = 0.2,
        first_detection_bonus: float = 0.2,
        portal_bonus: float = 0.2,
        agent_view_size: int = 3,
        **kwargs,
    ):
        assert target_style is None or target_style in self.VALID_STYLES, (
            f"target_style must be None or one of {self.VALID_STYLES}"
        )
        self.size = 13
        self.randomize_layout = randomize_layout
        self.end_on_detection = end_on_detection
        self.end_on_lava = end_on_lava
        self.detection_penalty = detection_penalty
        self.lava_penalty = lava_penalty
        self.pickup_bonus = pickup_bonus
        self.drop_penalty = drop_penalty
        self.kill_bonus = kill_bonus
        self.first_lava_bonus = first_lava_bonus
        self.first_detection_bonus = first_detection_bonus
        self.portal_bonus = portal_bonus

        self.target_style = target_style
        self.target_bonus = target_bonus
        self.non_target_penalty = non_target_penalty
        defaults = {s: 0.6 for s in self.VALID_STYLES}
        if style_bonuses:
            defaults.update(style_bonuses)
        self.style_bonuses = defaults

        # fixed extra detection tiles below the enemy (camo-guarded gap)
        self.extra_detection_tiles = {(9, 7), (10, 7)}

        # episode state
        self.enemy_pos: Optional[Tuple[int, int]] = None
        self.enemy_obj: Optional[Enemy] = None
        self.enemy_alive = True
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.portal_in_pos: Tuple[int, int] = (0, 0)
        self.portal_out_pos: Tuple[int, int] = (0, 0)

        self.killed_with_weapon = False
        self.traversed_detection = False
        self.traversed_lava = False
        self.used_portal = False
        self.detected = False
        self.died_in_lava = False
        self.style_used: Optional[str] = None

        self.weapon_picked = self.camo_picked = self.boots_picked = False
        self.weapon_dropped = self.camo_dropped = self.boots_dropped = False

        # control-metric counters
        self.step_count = 0
        self.min_distance_to_enemy = float("inf")
        self.sum_distance_to_enemy = 0.0
        self.forward_action_count = 0
        self.items_picked_count = 0
        self.detection_steps = 0
        self.lava_steps = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=max_steps or 4 * size * size,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "four_style_policy"

    # --- Grid construction -----------------------------------------------------

    def _gen_grid(self, width, height):
        # reset per-episode state
        self.killed_with_weapon = self.traversed_detection = self.traversed_lava = False
        self.used_portal = self.detected = self.died_in_lava = False
        self.style_used = None
        self.weapon_picked = self.camo_picked = self.boots_picked = False
        self.weapon_dropped = self.camo_dropped = self.boots_dropped = False
        self.step_count = 0
        self.min_distance_to_enemy = float("inf")
        self.sum_distance_to_enemy = 0.0
        self.forward_action_count = 0
        self.items_picked_count = 0
        self.detection_steps = 0
        self.lava_steps = 0

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # enemy (faces east) + goal
        self.enemy_pos = (6, 6)
        self.enemy_obj = Enemy(dir=0)
        self.put_obj(self.enemy_obj, *self.enemy_pos)
        self.enemy_alive = True
        self.goal_pos = (11, 6)
        self.put_obj(Goal(), *self.goal_pos)

        # top bar walls (with the camo-guarded gap at x9,10) and lava strip
        for x in (6, 7, 8):
            self.put_obj(Wall(), x, 7)
            self.put_obj(Wall(), x, 8)
        for y in range(9, 12):
            for x in range(6, 11):
                self.put_obj(HazardTile(), x, y)

        # portal exit is fixed by the goal
        self.portal_out_pos = (11, 3)

        # items: fixed cells, or random within per-item regions. The weapon and
        # camo regions match the three-style env; boots and portal-in keep their
        # own regions. Heads 2 / enemy / goal / agent stay fixed.
        if self.randomize_layout:
            occupied = {self.enemy_pos, self.goal_pos, self.portal_out_pos, (1, 8)}
            c_pos = self._rand_cell((1, 2), (2, 4), occupied)      # camo
            w_pos = self._rand_cell((3, 4), (3, 5), occupied)      # weapon
            b_pos = self._rand_cell((2, 3), (6, 7), occupied)      # boots
            self.portal_in_pos = self._rand_cell((2, 4), (11, 11), occupied)  # portal-in, always y=11
        else:
            c_pos, w_pos, b_pos = (1, 3), (3, 5), (3, 6)
            self.portal_in_pos = (3, 11)
        self.put_obj(Camouflage(), *c_pos)
        self.put_obj(Weapon(), *w_pos)
        self.put_obj(Boots(), *b_pos)
        self.put_obj(Portal(), *self.portal_in_pos)
        self.put_obj(Portal(color="green"), *self.portal_out_pos)

        # agent
        self.agent_dir = 0
        self.place_agent(top=(1, 8), size=(1, 1), rand_dir=False)

        self.mission = "Reach the goal via weapon, camouflage, daredevil, or portal."

    # --- Helpers ---------------------------------------------------------------

    def is_in_detection(self, pos) -> bool:
        if not self.enemy_alive or self.enemy_pos is None or tuple(pos) == self.goal_pos:
            return False
        ex, ey = self.enemy_pos
        ax, ay = pos
        in_box = (ex <= ax <= ex + 4) and (ey - 5 <= ay <= ey)
        return in_box or tuple(pos) in self.extra_detection_tiles

    @staticmethod
    def _is_adjacent(a, b) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _manhattan(a, b) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _rand_cell(xr, yr, occupied):
        """Random free cell within [xr]x[yr], avoiding `occupied` (mutated)."""
        while True:
            p = (random.randint(xr[0], xr[1]), random.randint(yr[0], yr[1]))
            if p not in occupied:
                occupied.add(p)
                return p

    def _agent_has_weapon(self) -> bool:
        return isinstance(self.carrying, Weapon)

    def _agent_has_camouflage(self) -> bool:
        return isinstance(self.carrying, Camouflage)

    def _agent_has_boots(self) -> bool:
        return isinstance(self.carrying, Boots)

    def _remove_enemy(self):
        if self.enemy_pos:
            self.grid.set(*self.enemy_pos, None)
        self.enemy_alive = False
        self.killed_with_weapon = True
        self.style_used = "weapon"

    def _achieved_style(self) -> Optional[str]:
        if self.killed_with_weapon:
            return "weapon"
        if self.traversed_detection:
            return "camouflage"
        if self.traversed_lava:
            return "daredevil"
        if self.used_portal:
            return "portal"
        return None  # goal unreachable without a mechanic; no plain bypass style

    # --- Step override ---------------------------------------------------------

    def step(self, action):
        prev_carrying = self.carrying
        obs, reward, terminated, truncated, info = super().step(action)

        if action == 2:
            self.forward_action_count += 1
        if action == 3:
            self.items_picked_count += 1

        if self.enemy_alive and self.enemy_pos is not None:
            d = self._manhattan(self.agent_pos, self.enemy_pos)
            self.min_distance_to_enemy = min(self.min_distance_to_enemy, d)
            self.sum_distance_to_enemy += d

        # pickup bonuses (first pickup of the target item only)
        if prev_carrying is None and self.carrying is not None:
            if isinstance(self.carrying, Weapon) and not self.weapon_picked:
                self.weapon_picked = True
                if self.target_style == "weapon":
                    reward += self.pickup_bonus
            elif isinstance(self.carrying, Camouflage) and not self.camo_picked:
                self.camo_picked = True
                if self.target_style == "camouflage":
                    reward += self.pickup_bonus
            elif isinstance(self.carrying, Boots) and not self.boots_picked:
                self.boots_picked = True
                if self.target_style == "daredevil":
                    reward += self.pickup_bonus

        # drop penalty (first drop of the target item after pickup)
        if prev_carrying is not None and self.carrying is None:
            if isinstance(prev_carrying, Weapon) and self.weapon_picked and not self.weapon_dropped:
                self.weapon_dropped = True
                if self.target_style == "weapon":
                    reward += self.drop_penalty
            elif isinstance(prev_carrying, Camouflage) and self.camo_picked and not self.camo_dropped:
                self.camo_dropped = True
                if self.target_style == "camouflage":
                    reward += self.drop_penalty
            elif isinstance(prev_carrying, Boots) and self.boots_picked and not self.boots_dropped:
                self.boots_dropped = True
                if self.target_style == "daredevil":
                    reward += self.drop_penalty

        # portal: Portal tiles are type 'lava', so super().step() flagged
        # termination — override it, and teleport from the entrance head.
        ap = tuple(self.agent_pos)
        if ap in (self.portal_in_pos, self.portal_out_pos):
            terminated = False
            if ap == self.portal_in_pos:
                self.agent_pos = self.portal_out_pos
                self.agent_dir = 1  # face south, toward the goal
                self.used_portal = True
                if self.target_style == "portal":
                    reward += self.portal_bonus
                obs = self.gen_obs()

        # lava
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, HazardTile):
            self.lava_steps += 1
            if self._agent_has_boots():
                if not self.traversed_lava and self.target_style == "daredevil":
                    reward += self.first_lava_bonus
                self.traversed_lava = True
            elif self.end_on_lava:
                self.died_in_lava = True
                info = dict(info)
                info["termination"] = "died_in_lava"
                info["detected"] = self.detected
                return obs, reward + self.lava_penalty, True, truncated, info

        # detection
        if self.is_in_detection(self.agent_pos):
            self.detection_steps += 1
            if self._agent_has_camouflage():
                if not self.traversed_detection and self.target_style == "camouflage":
                    reward += self.first_detection_bonus
                self.traversed_detection = True
            elif self.end_on_detection:
                self.detected = True
                info = dict(info)
                info["termination"] = "detected"
                info["detected"] = True
                return obs, reward + self.detection_penalty, True, truncated, info

        # weapon attack
        if action == self.actions.toggle and self.enemy_alive and self.enemy_pos is not None:
            if self._is_adjacent(self.agent_pos, self.enemy_pos) and self._agent_has_weapon():
                self._remove_enemy()
                if self.target_style == "weapon":
                    reward += self.kill_bonus

        # goal
        info = dict(info)
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Goal):
            terminated = True
            achieved = self._achieved_style()
            if self.target_style is None:
                bonus = self.style_bonuses.get(achieved, 0.0)
            else:
                bonus = self.target_bonus if achieved == self.target_style else self.non_target_penalty
            base = self._reward()
            reward = base + bonus
            avg_dist = self.sum_distance_to_enemy / max(self.step_count, 1) if self.enemy_alive else 0.0
            info["target_style"] = self.target_style
            info["achieved_style"] = achieved
            info["base_reward"] = base
            info["style_bonus_or_penalty"] = bonus
            info["total_reward"] = reward
            info["episode_summary"] = {
                "total_steps": self.step_count,
                "min_enemy_distance": self.min_distance_to_enemy if self.min_distance_to_enemy != float("inf") else 0.0,
                "avg_enemy_distance": avg_dist,
                "forward_steps": self.forward_action_count,
                "items_picked": self.items_picked_count,
                "path_efficiency": self.forward_action_count / max(self.step_count, 1),
                "was_detected": self.detected,
                "achieved_style": achieved,
                "killed_with_weapon": self.killed_with_weapon,
                "traversed_detection": self.traversed_detection,
                "traversed_lava": self.traversed_lava,
                "used_portal": self.used_portal,
                "detection_steps": self.detection_steps,
                "lava_steps": self.lava_steps,
            }
            return obs, reward, terminated, truncated, info

        info["detected"] = self.detected
        info["enemy_alive"] = self.enemy_alive
        info["step_count"] = self.step_count
        return obs, reward, terminated, truncated, info


# --- Registration helper -------------------------------------------------------

def register_env():
    gym.envs.registration.register(
        id="MiniGrid-FourStyles-v0",
        entry_point=MiniGridFourStyles,
    )


# --- Manual control ------------------------------------------------------------

if __name__ == "__main__":
    register_env()
    env = gym.make(
        "MiniGrid-FourStyles-v0",
        target_style="portal",
        target_bonus=1.0,
        non_target_penalty=-1.0,
        render_mode="human",
        randomize_layout=True,
        max_steps=100,
        agent_view_size=7,
    )
    obs, _ = env.reset()
    ret, finish = 0.0, False
    print("Controls: LEFT/RIGHT rotate, UP forward, SPACE pickup, TAB drop, "
          "DOWN toggle, Z done, R reset")
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True
                break
            if event.type == pygame.KEYDOWN:
                action = {pygame.K_LEFT: 0, pygame.K_RIGHT: 1, pygame.K_UP: 2,
                          pygame.K_SPACE: 3, pygame.K_TAB: 4, pygame.K_DOWN: 5,
                          pygame.K_z: 6}.get(event.key)
                if event.key == pygame.K_r:
                    obs, _ = env.reset(); ret = 0.0; print("--- reset ---"); continue
                if action is not None:
                    obs, reward, done, trunc, info = env.step(action)
                    ret += reward
                    print(f"info: {info}")
                    print(f"step r: {reward:.3f}  return: {ret:.3f}")
                    if done or trunc:
                        print(f"*** end (term={done}, trunc={trunc}) return={ret:.3f} ***")
                        finish = True
                        break
        if finish:
            break
        env.render()
        env.unwrapped.clock.tick(10)
    env.close()