from __future__ import annotations
from typing import Optional, Tuple
import random

import gymnasium as gym
import pygame
from gymnasium import spaces
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Goal, Wall, Ball, Box, Lava
from minigrid.core.grid import Grid
from minigrid.core.constants import DIR_TO_VEC
import numpy as np
from envs.metalgridsolid.utils.utils import draw_triangle
from envs.old.minigrid_dungeon.manual_control import ManualControl


# --- rendering utils -----------------------------------------------------------
def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn

def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


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


    def render(self, img):
        c = np.array([155, 89, 182])

        def create_enemy_mask(dimensions):
            """
            Create a mask for a sword shape based on normalized coordinates (from 0 to 1).
            """
            # Central position based on dimensions
            cx, cy = 0.5, 0.5  # Center in normalized coordinates
            radius = 0.2  # Normalized radius of the central ball
            num_spikes = 10  # Total number of spikes
            spike_length = 0.3  # Normalized length of spikes
            spike_width = 0.1  # Normalized width of each spike

            def spiky_ball_fn(x, y):
                # Check if inside the main ball
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    return True

                # Check each spike
                for i in range(num_spikes):
                    angle = 2 * np.pi * i / num_spikes
                    # Define the start of the spike at the edge of the ball
                    spike_base_x = cx + radius * np.cos(angle)
                    spike_base_y = cy + radius * np.sin(angle)
                    spike_tip_x = cx + (radius + spike_length) * np.cos(angle)
                    spike_tip_y = cy + (radius + spike_length) * np.sin(angle)

                    # Determine if point (x, y) is within the boundaries of this spike
                    # Calculate distance to the line defined by the spike
                    dx = spike_tip_x - spike_base_x
                    dy = spike_tip_y - spike_base_y
                    length = np.sqrt(dx ** 2 + dy ** 2)
                    if length == 0:
                        continue  # Prevent division by zero
                    dx /= length
                    dy /= length
                    p = (x - spike_base_x) * dx + (y - spike_base_y) * dy
                    if 0 <= p <= length:  # Check if point is along the spike's length
                        dist = abs((x - spike_base_x) * dy - (y - spike_base_y) * dx)
                        if dist <= spike_width / 2:  # Check if point is within the width
                            return True

                return False

            return spiky_ball_fn

        mask_fn = create_enemy_mask(img.shape)
        fill_coords(img, mask_fn, c)


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
        c = np.array([241, 196, 15])

        def create_sword_mask(dimensions):
            """
            Create a mask for a sword shape based on normalized coordinates (from 0 to 1).
            """
            cx, cy = 0.5, 0.8  # Center x and lower center y for the sword, adjusted for visual balance
            blade_height = 0.7
            handle_height = 0.1
            blade_width = 0.1
            handle_width = 0.3
            pommel_radius = 0.07

            def sword_fn(x, y):
                # Blade area
                blade_top = cy - blade_height
                blade_left = cx - blade_width / 2
                blade_right = cx + blade_width / 2
                if blade_top <= y <= cy and blade_left <= x <= blade_right:
                    return True

                # Handle area
                handle_bottom = cy
                handle_top = handle_bottom - handle_height
                handle_left = cx - handle_width / 2
                handle_right = cx + handle_width / 2
                if handle_top <= y <= handle_bottom and handle_left <= x <= handle_right:
                    return True

                # Pommel area (circle at the bottom of the handle)
                if (x - cx) ** 2 + (y - (handle_bottom + pommel_radius)) ** 2 <= pommel_radius ** 2:
                    return True

                return False

            return sword_fn

        mask_fn = create_sword_mask(img.shape)
        fill_coords(img, mask_fn, c)


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
        fill_coords(img, lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.35 ** 2, (34, 85, 34))


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
            randomize_layout: bool = False,  # randomize positions for trajectory diversity
            layout: str = "default",  # "default" | "large"
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
        self.randomize_layout = randomize_layout
        self.layout = layout
        if layout == "large":
            self.size = 15
        elif not self.easy_env:
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

        # Tracking variables for control metric computation
        self.step_count = 0
        self.min_distance_to_enemy = float('inf')
        self.sum_distance_to_enemy = 0.0
        self.forward_action_count = 0
        self.items_picked_count = 0

        self.grid = self._create_grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.layout == "large":
            self._gen_large_layout(width, height)
            return

        # Layout:
        # Agent starts on the left, goal on the far right.
        # Enemy approximately in the middle, facing right.
        # Weapon somewhere else (top-left-ish).
        # Keep things open so bypass vs backstab vs weapon are all feasible.

        # Place goal
        if self.randomize_layout:
            goal_x = width - 2
            goal_y = random.randint(height // 2 , height // 2 + 1)
        else:
            goal_x, goal_y = width - 2, height // 2
        self.goal_pos = (goal_x, goal_y)
        self.put_obj(Goal(), goal_x, goal_y)

        # Place enemy with randomization
        # if self.randomize_layout:
        #     if self.easy_env:
        #         ex = random.randint(width // 2 - 1, width // 2 + 1)
        #         ey = random.randint(height // 2 - 1, height // 2 + 1)
        #     else:
        #         ex = random.randint(width // 2 , width // 2 )
        #         ey = random.randint(height // 2 - 2, height // 2 )
        # else:
        ex, ey = width // 2, height // 2
        self.enemy_dir = 0  # facing right

        self.enemy_obj = Enemy(dir=self.enemy_dir)
        self.put_obj(self.enemy_obj, ex, ey)
        self.enemy_pos = (ex, ey)
        self.enemy_alive = True

        # Place weapon with randomization
        if self.randomize_layout:
            if self.easy_env:
                wx = random.randint(2, 3)
                wy = random.randint(1, 2)
            else:
                wx = random.randint(3, 4)
                wy = random.randint(3, 5)
        else:
            if self.easy_env:
                wx, wy = 2, 1
            else:
                wx, wy = 3, 5
        self.put_obj(Weapon(), wx, wy)

        # Place Camouflage with randomization
        if not self.easy_env:
            if self.randomize_layout:
                occupied = {(ex, ey), (wx, wy)}
                while True:
                    cx = random.randint(1, 2)
                    cy = random.randint(2, 4)
                    if (cx, cy) not in occupied:
                        break
            else:
                cx, cy = 1, 3
            self.put_obj(Camouflage(), cx, cy)

        # Optionally add some walls to make routing interesting but not forced
        # (Keep simple; you can tweak as needed)
        # Example: a little pillar above the enemy to shape paths
        if not self.easy_env:
            px, py = width // 2, height // 2 + 1
            if 1 < py < height - 1:
                for i in range(4):
                    self.put_obj(Wall(), px + i, py)
                    self.put_obj(Wall(), px + i, py + 2)
                # self.put_obj(Wall(), px, py + 1)

        # Place agent with randomization
        if self.randomize_layout:
            if self.easy_env:
                agent_y = random.randint(height // 2 + 1, height // 2 + 3)
            else:
                agent_y = random.randint(height // 2, height // 2 + 3)
            self.place_agent(top=(1, agent_y), size=(1, 1))
        else:
            self.place_agent(top=(1, height // 2 + 2), size=(1, 1))

        self.detected = False
        self.style_used = None
        self.mission = (
            "Reach the goal via one of three styles: "
            "(1) bypass unseen, (2) backstab from behind with toggle, "
            "(3) pick up weapon and toggle to defeat enemy."
        )


    def _gen_large_layout(self, width, height):
        """
        Large 15×15 layout with two horizontal tunnels and scattered obstacles.
        Every style has multiple genuinely distinct paths.

        Fixed layout (x→, y↓):
          y=2  . . [####] . . . . . . .    north tunnel top wall, x=4..8
          y=3  . . [ W  . . ] . . . . .    north tunnel interior; weapon at (5,3)
          y=4  . . [## . ##] . . . . . .   north tunnel bottom wall, gap at x=6
          y=5  . # . . . . . . . . . . .   obstacle (3,5)
          y=6  . . . . . . # . . . . . .   central obstacle at (7,6)
          y=7  A . . . . . # . . E . . .   agent, central obstacle (7,7), enemy (10,7)
          y=8  . . . . . . . . # . . . .   mid-right obstacle (8,8)
          y=9  . # . . . . . . . . ##. .   (3,9), near-goal (11,9)(12,9)
          y=10 . . [## . ##] . . . . . .   south tunnel top wall, gap at x=6
          y=11 . . [ . C  . ] . . . . .    south tunnel interior; camouflage at (5,11)
          y=12 . . [####] . . . . . . .    south tunnel bottom wall, x=4..8
          Goal at (13,9).  Enemy detection: x:10-14, y:2-7 (non-easy).

        Multiple paths per style (all verified):
          bypass     — y=1 corridor east to x=9, south to y=8, east to goal  (A)
                       south tunnel (y=11) east to exit, north to y=8, east to goal  (B)
                       middle zigzag at y=8 via detours around (8,8) and (11,9)     (C)
          backstab   — exit north tunnel (9,3) → south to (9,7), toggle  (A)
                       exit south tunnel (9,11) → north to (9,7), toggle  (B)
          weapon     — enter north tunnel from west at (3,3)  or
                       from north (x≤3, y=3) then east  or
                       from south via gap at (6,4) → west to weapon;
                       attack from (9,7) [behind] or (10,8) [below enemy]
          camouflage — enter detection zone at y=2..3 (top edge route)
                       enter at y=6..7 coming from below (south-side route)
        """
        if self.randomize_layout:
            goal_x = width - 2
            goal_y = random.randint(8, 10)
            ex = random.randint(9, 11)
            ey = random.randint(6, 8)
            wx = random.randint(4, 7)
            wy = 3   # always inside north tunnel
            agent_y = random.randint(6, 8)
        else:
            goal_x, goal_y = width - 2, 9
            ex, ey = 10, 7
            wx, wy = 5, 3
            agent_y = 7

        # Place goal
        self.goal_pos = (goal_x, goal_y)
        self.put_obj(Goal(), goal_x, goal_y)

        # Place enemy — always faces right in this layout
        self.enemy_obj = Enemy(dir=0)
        self.put_obj(self.enemy_obj, ex, ey)
        self.enemy_pos = (ex, ey)
        self.enemy_alive = True

        # Place weapon inside north tunnel at y=3
        self.put_obj(Weapon(), wx, wy)

        # Place camouflage inside south tunnel at y=11 (non-easy only).
        # wy=3 and cy=11 are different rows so no position collision is possible.
        if not self.easy_env:
            cx = random.randint(4, 7) if self.randomize_layout else 5
            cy = 11
            self.put_obj(Camouflage(), cx, cy)

        # Place agent between the two tunnels
        self.place_agent(top=(1, agent_y), size=(1, 1))

        # --- Walls ---

        # North tunnel top wall (solid) — forms ceiling of weapon tunnel
        for i in range(4, 9):
            self.put_obj(Wall(), i, 2)

        # North tunnel bottom wall — gap at x=6 allows south entry into tunnel
        for i in range(4, 9):
            if i != 6:
                self.put_obj(Wall(), i, 4)

        # South tunnel top wall — gap at x=6 allows north entry into tunnel
        for i in range(4, 9):
            if i != 6:
                self.put_obj(Wall(), i, 10)

        # South tunnel bottom wall (solid) — forms floor of camouflage tunnel
        for i in range(4, 9):
            self.put_obj(Wall(), i, 12)

        # Central vertical obstacle — forces routing north (y≤5) or south (y≥8)
        # when approaching the enemy from the west; creates two attack lanes
        self.put_obj(Wall(), 7, 6)
        self.put_obj(Wall(), 7, 7)

        # Mid-right obstacle — breaks the trivial straight east path at y=8
        self.put_obj(Wall(), 8, 8)

        # Left-centre obstacles — push agents toward one tunnel or the other
        self.put_obj(Wall(), 3, 5)
        self.put_obj(Wall(), 3, 9)

        # Near-goal obstacles — force a final routing choice at the goal approach
        self.put_obj(Wall(), 11, 9)
        self.put_obj(Wall(), 12, 9)

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

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Compute Manhattan distance between two positions."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

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
        # reward -= 0.001  # step penalty: comment out for Bypass mode or camouflage longer paths

        # Track metrics for control computation
        # self.step_count += 1

        # Track distance to enemy
        if self.enemy_alive and self.enemy_pos is not None:
            dist = self._manhattan_distance(self.agent_pos, self.enemy_pos)
            self.min_distance_to_enemy = min(self.min_distance_to_enemy, dist)
            self.sum_distance_to_enemy += dist

        # Track action types
        if action == 2:  # forward action
            self.forward_action_count += 1
        if action == 3:  # pickup action
            self.items_picked_count += 1

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

            # Add episode summary for control metric computation
            avg_distance = self.sum_distance_to_enemy / max(self.step_count, 1) if self.enemy_alive else 0.0
            info["episode_summary"] = {
                "total_steps": self.step_count,
                "min_enemy_distance": self.min_distance_to_enemy if self.min_distance_to_enemy != float('inf') else 0.0,
                "avg_enemy_distance": avg_distance,
                "forward_steps": self.forward_action_count,
                "items_picked": self.items_picked_count,
                "path_efficiency": self.forward_action_count / max(self.step_count, 1),
                "was_detected": self.detected,
                "achieved_style": achieved,
                "picked_weapon": self.weapon_picked,
                "picked_camouflage": self.camouflage_picked,
            }

            return obs, reward, terminated, truncated, info


        info["style"] = self.style_used
        info["detected"] = self.detected
        info["enemy_alive"] = self.enemy_alive

        # Add per-step metrics
        if self.enemy_pos is not None:
            info["distance_to_enemy"] = self._manhattan_distance(self.agent_pos, self.enemy_pos)
        info["step_count"] = self.step_count

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
                   render_mode="human", easy_env=False, max_steps=100, randomize_layout=True, layout="default")
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
