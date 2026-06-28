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

class Enemy(WorldObj):
    """Stationary enemy with a facing direction. Detection logic lives in the env."""

    def __init__(self, color="blue", dir=0):
        super().__init__("box", color)
        self.dir = dir

    def can_overlap(self):
        return False

    def render(self, img):
        c = np.array([155, 89, 182])
        cx, cy = 0.5, 0.5
        radius = 0.2
        num_spikes = 10
        spike_length = 0.3
        spike_width = 0.1

        def spiky_ball_fn(x, y):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                return True
            for i in range(num_spikes):
                angle = 2 * np.pi * i / num_spikes
                base_x = cx + radius * np.cos(angle)
                base_y = cy + radius * np.sin(angle)
                tip_x = cx + (radius + spike_length) * np.cos(angle)
                tip_y = cy + (radius + spike_length) * np.sin(angle)
                dx, dy = tip_x - base_x, tip_y - base_y
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length == 0:
                    continue
                dx /= length
                dy /= length
                p = (x - base_x) * dx + (y - base_y) * dy
                if 0 <= p <= length:
                    dist = abs((x - base_x) * dy - (y - base_y) * dx)
                    if dist <= spike_width / 2:
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
        cx, cy = 0.5, 0.8
        blade_h, handle_h = 0.7, 0.1
        blade_w, handle_w = 0.1, 0.3
        pommel_r = 0.07

        def sword_fn(x, y):
            if cy - blade_h <= y <= cy and cx - blade_w / 2 <= x <= cx + blade_w / 2:
                return True
            if cy - handle_h <= y <= cy and cx - handle_w / 2 <= x <= cx + handle_w / 2:
                return True
            if (x - cx) ** 2 + (y - (cy + pommel_r)) ** 2 <= pommel_r ** 2:
                return True
            return False

        fill_coords(img, sword_fn, c)


class Camouflage(WorldObj):
    def __init__(self, color="red"):
        # Reuse the otherwise-unused 'door' type so this item gets a distinct
        # object index (obs channel 0) from Weapon/Boots. Behaviour comes from
        # the methods below, not the type string: it subclasses WorldObj (not
        # Door), so encode()/can_overlap()/toggle() are the inert defaults, and
        # the env's style logic identifies it via isinstance(Camouflage).
        super().__init__("door", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(
            img,
            lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.35 ** 2,
            (34, 85, 34),
        )


class Boots(WorldObj):
    """Boots; while carried, the agent can traverse lava tiles safely."""

    def __init__(self, color="purple"):
        # Reuse the otherwise-unused 'floor' type for a distinct object index
        # (obs channel 0) from Weapon/Camouflage. See Camouflage for why the
        # type string is purely a label and changes no behaviour here.
        super().__init__("floor", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = np.array([90, 60, 30])
        # leg
        fill_coords(img, point_in_rect(0.38, 0.55, 0.18, 0.7), c)
        # foot (extending forward)
        fill_coords(img, point_in_rect(0.30, 0.80, 0.70, 0.85), c)
        # sole highlight
        fill_coords(img, point_in_rect(0.30, 0.80, 0.83, 0.88), (40, 25, 10))


class HazardTile(WorldObj):
    """Lava-like hazard. Steps onto it: instant death unless carrying Boots.

    Uses type='ball' (rather than 'lava') so MiniGrid's built-in lava-death
    handling does not fire; we handle it ourselves in step().
    """

    def __init__(self):
        super().__init__("ball", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), (255, 100, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.25, 0.35), (200, 60, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.55, 0.65), (200, 60, 0))
        fill_coords(img, point_in_rect(0.0, 1.0, 0.85, 0.95), (200, 60, 0))


class DetectionMarker(WorldObj):
    """Visual-only marker for detection zones.

    Has no gameplay effect: can be walked over freely, and the env's detection
    logic compares the agent's position against the static detection boxes
    (not against marker presence). Only added to the grid when
    `show_detection_zones=True` so default (training) obs are unchanged.
    """

    def __init__(self):
        super().__init__("ball", "purple")

    def can_overlap(self):
        return True

    def render(self, img):
        # tinted floor — distinctive but readable under other rendering
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), (120, 50, 60))


# --- Environment ---------------------------------------------------------------

class MiniGridMultiStyles(MiniGridEnv):
    """
    17x17 multi-style MiniGrid environment with four distinct success styles:

      bypass     - reach goal via the outer perimeter, no items used, no
                   detection, no lava.
      weapon     - pick up a weapon, kill an enemy, reach goal.
      camouflage - pick up a camouflage, walk through an enemy detection zone,
                   reach goal.
      daredevil  - pick up boots, walk through the lava strip, reach goal.

    Achievement priority (in case multiple conditions are satisfied):
      weapon > camouflage > daredevil > bypass.

    Layout (17x17, x=col, y=row, top-left origin). Detection zones are
    forward-facing cones only (no detection behind the enemy).

           x: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
     y= 0:    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
     y= 1:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y= 2:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y= 3:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y= 4:    #  .  #  #  #  #  #  #  #  #  #  #  #  #  #  .  #
     y= 5:    #  .  #  B  B  .  .  W  W  .  .  d  d  d  .  .  #
     y= 6:    #  .  #  B  B  C  C  W  W  .  .  E  d  d  #  .  #
     y= 7:    #  .  #  .  .  C  C  .  .  .  .  d  d  d  #  .  #
     y= 8:    #  A  .  .  .  .  .  L  L  L  L  L  L  L  .  G  #
     y= 9:    #  .  #  .  .  C  C  .  .  .  .  d  d  d  #  .  #
     y=10:    #  .  #  B  B  C  C  W  W  .  .  E  d  d  #  .  #
     y=11:    #  .  #  B  B  .  .  W  W  .  .  d  d  d  .  .  #
     y=12:    #  .  #  #  #  #  #  #  #  #  #  #  #  #  #  .  #
     y=13:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y=14:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y=15:    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  #
     y=16:    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

    Legend: W=weapon, C=camouflage, B=boots, E=enemy, d=detection zone tile,
    L=lava, A=agent start, G=goal.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    VALID_STYLES = ("bypass", "weapon", "camouflage", "daredevil")

    def __init__(
        self,
        size: int = 17,
        max_steps: Optional[int] = None,
        *,
        target_style: Optional[str] = None,
        target_bonus: float = 0.6,
        non_target_penalty: float = 0.0,
        style_bonuses: Optional[dict] = None,
        randomize_layout: bool = False,
        free_item_placement: bool = False,
        lava_penalty: float = -0.3,
        detection_penalty: float = -0.5,
        end_on_detection: bool = True,
        end_on_lava: bool = True,
        shape_rewards: bool = True,
        bypass_step_penalty: float = -0.004,
        bypass_corridor: Optional[str] = None,
        wrong_corridor_penalty: float = -0.5,
        style_step_penalty: float =  -0.004, # -0.004,
        pickup_bonus: float = 0.2,
        drop_penalty: float = -0.2,
        kill_bonus: float = 0.4,
        exit_approach_coef: float = 0.05,
        first_detection_bonus: float = 0.3,
        first_lava_bonus: float = 0.3,
        show_detection_zones: bool = False,
        agent_view_size: int = 3,
        **kwargs,
    ):
        assert target_style is None or target_style in self.VALID_STYLES, (
            f"target_style must be None or one of {self.VALID_STYLES}"
        )
        assert bypass_corridor in (None, "upper", "lower"), (
            "bypass_corridor must be None, 'upper', or 'lower'"
        )

        self.bypass_corridor = bypass_corridor
        self.wrong_corridor_penalty = wrong_corridor_penalty
        self.size = size
        self.agent_view_size = agent_view_size
        self.randomize_layout = randomize_layout
        self.free_item_placement = free_item_placement
        self.lava_penalty = lava_penalty
        self.detection_penalty = detection_penalty
        self.end_on_detection = end_on_detection
        self.end_on_lava = end_on_lava
        self.show_detection_zones = show_detection_zones

        # reward-shaping configuration
        self.shape_rewards = shape_rewards
        self.bypass_step_penalty = bypass_step_penalty
        self.style_step_penalty = style_step_penalty
        self.pickup_bonus = pickup_bonus
        self.drop_penalty = drop_penalty
        self.kill_bonus = kill_bonus
        self.exit_approach_coef = exit_approach_coef
        self.first_detection_bonus = first_detection_bonus
        self.first_lava_bonus = first_lava_bonus

        # per-episode shaping flags (reset in _gen_grid)
        self.weapon_picked = False
        self.weapon_dropped = False
        self.camo_picked = False
        self.camo_dropped = False
        self.boots_picked = False
        self.boots_dropped = False
        self.first_detection_bonus_paid = False
        self.first_lava_bonus_paid = False

        self.target_style = target_style
        self.target_bonus = target_bonus
        self.non_target_penalty = non_target_penalty

        default_bonus = 0.6
        defaults = {s: default_bonus for s in self.VALID_STYLES}
        if style_bonuses:
            defaults.update(style_bonuses)
        self.style_bonuses = defaults

        # episode state (re-initialised in _gen_grid each reset)
        self.enemy1_obj: Optional[Enemy] = None
        self.enemy1_pos: Optional[Tuple[int, int]] = None
        self.enemy1_alive: bool = True
        self.enemy1_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.enemy2_obj: Optional[Enemy] = None
        self.enemy2_pos: Optional[Tuple[int, int]] = None
        self.enemy2_alive: bool = True
        self.enemy2_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)

        self.killed_with_weapon = False
        self.traversed_detection_with_camo = False
        self.traversed_lava_with_boots = False
        self.detected = False
        self.died_in_lava = False
        self.style_used: Optional[str] = None

        # tracking metrics
        self.step_count = 0
        self.min_distance_to_enemy = float("inf")
        self.sum_distance_to_enemy = 0.0
        self.forward_action_count = 0
        self.items_picked_count = 0

        # behavioural counters for continuous control variables
        self.lava_steps = 0
        self.lava_adjacent_steps = 0
        self.enemy_adjacent_steps = 0
        self.enemy_near_unprotected_steps = 0
        self.detection_cone_steps = 0

        # post-kill exit shaping state
        self._prev_exit_dist: Optional[float] = None
        self._exit_phase: Optional[str] = None
        self._exit_wp_idx: int = 0

        # bypass perimeter-shaping state
        self._bypass_wp_idx: int = 0
        self._prev_bypass_dist: Optional[float] = None

        # daredevil boots->lava->goal shaping state
        self._boots_positions: list = []
        self._prev_dd_dist: Optional[float] = None
        self._dd_phase: Optional[str] = None

        # weapon pre-kill pickup->enemy shaping state
        self._weapon_positions: list = []
        self._prev_wpn_dist: Optional[float] = None
        self._wpn_phase: Optional[str] = None

        # camouflage pre-traversal pickup->cone shaping state
        self._camo_positions: list = []
        self._prev_camo_dist: Optional[float] = None
        self._camo_phase: Optional[str] = None

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
        return "multi_style_policy"

    # --- Grid construction -----------------------------------------------------

    def _gen_grid(self, width, height):
        # reset all per-episode state
        self.killed_with_weapon = False
        self.traversed_detection_with_camo = False
        self.traversed_lava_with_boots = False
        self.detected = False
        self.died_in_lava = False
        self.style_used = None

        # shaping flags
        self.weapon_picked = False
        self.weapon_dropped = False
        self.camo_picked = False
        self.camo_dropped = False
        self.boots_picked = False
        self.boots_dropped = False
        self.first_detection_bonus_paid = False
        self.first_lava_bonus_paid = False

        self.step_count = 0
        self.min_distance_to_enemy = float("inf")
        self.sum_distance_to_enemy = 0.0
        self.forward_action_count = 0
        self.items_picked_count = 0

        # behavioural counters for continuous control variables
        self.lava_steps = 0
        self.lava_adjacent_steps = 0
        self.enemy_adjacent_steps = 0
        self.enemy_near_unprotected_steps = 0
        self.detection_cone_steps = 0

        # post-kill exit shaping state
        self._prev_exit_dist = None
        self._exit_phase = None
        self._exit_wp_idx = 0

        # bypass perimeter-shaping state
        self._bypass_wp_idx = 0
        self._prev_bypass_dist = None

        # daredevil boots->lava->goal shaping state
        self._boots_positions = []
        self._prev_dd_dist = None
        self._dd_phase = None

        self._weapon_positions = []
        self._prev_wpn_dist = None
        self._wpn_phase = None

        self._camo_positions = []
        self._prev_camo_dist = None
        self._camo_phase = None

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- chamber walls (closed room, three east exits) -------------------
        # north wall: row 4, cols 2..14
        for x in range(2, 15):
            self.put_obj(Wall(), x, 4)
        # south wall: row 13, cols 2..14
        for x in range(2, 15):
            self.put_obj(Wall(), x, 12)
        # west wall: col 2, rows 4..13 (opening at row 8 for agent entry)
        for y in range(4, 13):
            if y != 8:
                self.put_obj(Wall(), 2, y)
        # east wall: col 14, rows 4..13. Openings at rows 5 (camo N exit),
        # 8 (lava exit / daredevil), 11 (camo S exit).
        for y in range(4, 13):
            if y not in (5, 8, 11):
                self.put_obj(Wall(), 14, y)

        # --- lava strip on row 8, cols 7..13 ---------------------------------
        for x in range(7, 14):
            self.put_obj(HazardTile(), x, 8)

        # --- enemies + detection zones --------------------------------------
        # Detection boxes are forward-facing cones only (no detection behind
        # the enemy). Size is preserved by extending forward, not adding cells.
        # E1 (north chamber): faces east (dir=0). Front-only 3x3 cone, cols
        # 11..13 rows 5..7.
        self.enemy1_pos = (11, 6)
        self.enemy1_box = (11, 13, 5, 7)  # x_min, x_max, y_min, y_max
        self.enemy1_obj = Enemy(dir=0)
        self.put_obj(self.enemy1_obj, *self.enemy1_pos)
        self.enemy1_alive = True

        # E2 (south chamber): faces east (dir=0). Front-only 3x4 cone, cols
        # 11..13 rows 9..12 (enemy column + 2 forward).
        self.enemy2_pos = (11, 10)
        self.enemy2_box = (11, 13, 9, 11)
        self.enemy2_obj = Enemy(dir=0)
        self.put_obj(self.enemy2_obj, *self.enemy2_pos)
        self.enemy2_alive = True

        # --- items (with optional small randomisation) -----------------------
        # Boots are placed off the east-west axis so daredevil requires a
        # detour rather than a straight-line lava traversal.
        def pick(choices, default):
            return random.choice(choices) if self.randomize_layout else default

        if self.free_item_placement:
            # Dedicated per-item zones: a 2x2 square top and bottom (mirror-
            # symmetric across the lava row). Each item spawns in a random cell
            # of its own zone, so styles stay spatially separable while keeping
            # top/bottom route diversity. Columns are disjoint (boots x3-4,
            # camo x5-6, weapon x7-8), so items never collide.
            boots_cells  = [(3, 5), (4, 5), (3, 6), (4, 6),
                            (3, 10), (4, 10), (3, 11), (4, 11)]
            camo_top     = [(5, 6), (6, 6), (5, 7), (6, 7)]
            camo_bot     = [(5, 9), (6, 9), (5, 10), (6, 10)]
            weapon_top   = [(7, 5), (8, 5), (7, 6), (8, 6)]
            weapon_bot   = [(7, 10), (8, 10), (7, 11), (8, 11)]

            # Only when training that style: restrict its TARGET item to the
            # committed corridor side so it never sits across the lava from the
            # target enemy/cone (which would force a deadly crossing). Otherwise
            # (item is just a distractor, or no side) it can be either top/bottom.
            def _side_cells(style, top, bot):
                if self.target_style == style and self.bypass_corridor == "upper":
                    return top
                if self.target_style == style and self.bypass_corridor == "lower":
                    return bot
                return top + bot

            weapon_cells = _side_cells("weapon", weapon_top, weapon_bot)
            camo_cells = _side_cells("camouflage", camo_top, camo_bot)

            w_pos = random.choice(weapon_cells)
            c_pos = random.choice(camo_cells)
            b_pos = random.choice(boots_cells)
            self.put_obj(Weapon(), *w_pos)
            self.put_obj(Camouflage(), *c_pos)
            self.put_obj(Boots(), *b_pos)
            self._boots_positions = [b_pos]
            self._weapon_positions = [w_pos]
            self._camo_positions = [c_pos]
        else:
            w1 = pick([(7, 5), (8, 5)], (7, 5))
            w2 = pick([(3, 11), (4, 11)], (3, 11))
            c1 = pick([(5, 5), (6, 5)], (5, 5))
            c2 = pick([(5, 11), (6, 11)], (5, 11))
            b1 = pick([(3, 6), (4, 6)], (3, 6))
            b2 = pick([(8, 11), (8, 12)], (8, 11))

            self.put_obj(Weapon(), *w1)
            self.put_obj(Weapon(), *w2)
            self.put_obj(Camouflage(), *c1)
            self.put_obj(Camouflage(), *c2)
            self.put_obj(Boots(), *b1)
            self.put_obj(Boots(), *b2)
            self._boots_positions = [b1, b2]
            self._weapon_positions = [w1, w2]
            self._camo_positions = [c1, c2]

        # --- goal ------------------------------------------------------------
        self.goal_pos = (15, 8)
        self.put_obj(Goal(), *self.goal_pos)

        # --- agent -----------------------------------------------------------
        self.agent_dir = 0  # facing east
        self.place_agent(top=(1, 8), size=(1, 1), rand_dir=False)

        # --- optional detection-zone visualisation --------------------------
        # Only fills empty cells inside the detection boxes so existing
        # objects (enemy, treasure) keep their own rendering. Markers are
        # also skipped under the agent.
        if self.show_detection_zones:
            for box in (self.enemy1_box, self.enemy2_box):
                x_min, x_max, y_min, y_max = box
                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        if (x, y) == tuple(self.agent_pos):
                            continue
                        if self.grid.get(x, y) is None:
                            self.put_obj(DetectionMarker(), x, y)

        self.mission = (
            "Reach the goal via one of four styles: "
            "bypass via outer perimeter, kill an enemy with a weapon, "
            "walk through detection while holding camouflage, "
            "or walk through lava while holding boots."
        )

    # --- Helpers ---------------------------------------------------------------

    def _is_in_box(self, pos, box) -> bool:
        x_min, x_max, y_min, y_max = box
        x, y = pos
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _is_in_any_detection(self, pos) -> bool:
        if pos == self.goal_pos:
            return False
        in1 = self.enemy1_alive and self._is_in_box(pos, self.enemy1_box)
        in2 = self.enemy2_alive and self._is_in_box(pos, self.enemy2_box)
        return in1 or in2

    @staticmethod
    def _is_adjacent(a, b) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _manhattan(a, b) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _agent_has_weapon(self) -> bool:
        return self.carrying is not None and isinstance(self.carrying, Weapon)

    def _agent_has_camouflage(self) -> bool:
        return self.carrying is not None and isinstance(self.carrying, Camouflage)

    def _agent_has_boots(self) -> bool:
        return self.carrying is not None and isinstance(self.carrying, Boots)

    def _kill_enemy(self, idx: int):
        if idx == 1 and self.enemy1_alive:
            self.grid.set(*self.enemy1_pos, None)
            self.enemy1_alive = False
            self.killed_with_weapon = True
            self.style_used = "weapon"
        elif idx == 2 and self.enemy2_alive:
            self.grid.set(*self.enemy2_pos, None)
            self.enemy2_alive = False
            self.killed_with_weapon = True
            self.style_used = "weapon"

    def _weapon_target_enemy(self) -> Optional[int]:
        """Which enemy counts for the weapon style this run.

        Driven by `bypass_corridor`: 'upper' -> E1 (top), 'lower' -> E2
        (bottom), None -> either enemy counts (default). When a side is set,
        only the target enemy can be killed for credit, so the only path to
        positive return is to kill that enemy. The weapon may still spawn on
        either side and the agent may roam anywhere; nothing about the layout
        changes.
        """
        if self.bypass_corridor == "upper":
            return 1
        if self.bypass_corridor == "lower":
            return 2
        return None

    def _weapon_exit_gate(self) -> Optional[Tuple[int, int]]:
        """The east-wall gate cell the agent must exit through for a
        side-specific weapon run: 'upper' -> row-5 gate, 'lower' -> row-11
        gate. None when no side is set (no post-kill exit shaping)."""
        if self.bypass_corridor == "upper":
            return (14, 5)
        if self.bypass_corridor == "lower":
            return (14, 11)
        return None

    def _in_camo_target_detection(self, pos) -> bool:
        """Whether pos is inside the camouflage *target* detection zone.

        Mirrors `_weapon_target_enemy` for the camouflage style: driven by
        `bypass_corridor`, 'upper' -> E1 (top) zone, 'lower' -> E2 (bottom)
        zone, None -> either active zone counts (default). When a side is set,
        only traversing the matching zone with camo earns camouflage credit, so
        the only positive-return route is that side — biasing the run purely
        through the reward, with the layout unchanged.
        """
        if self.bypass_corridor == "upper":
            return self.enemy1_alive and self._is_in_box(pos, self.enemy1_box)
        if self.bypass_corridor == "lower":
            return self.enemy2_alive and self._is_in_box(pos, self.enemy2_box)
        return self._is_in_any_detection(pos)

    def _bypass_waypoints(self):
        """Ordered waypoints along the safe outer ring to the goal, per side.

        The bypass route hugs the outer corridor (col 1 / row 1 or 15 / col 15)
        to avoid both detection cones and the lava strip, so a direct
        distance-to-goal potential would wrongly pull the agent into the
        chamber. 'upper' -> north ring, 'lower' -> south ring. None -> no
        shaping (no committed side)."""
        if self.bypass_corridor == "upper":
            return [(1, 1), (15, 1), self.goal_pos]
        if self.bypass_corridor == "lower":
            return [(1, 15), (15, 15), self.goal_pos]
        return None

    def _achieved_style(self) -> str:
        # priority: weapon > camouflage > daredevil > bypass
        if self.killed_with_weapon:
            return "weapon"
        if self.traversed_detection_with_camo:
            return "camouflage"
        if self.traversed_lava_with_boots:
            return "daredevil"
        return "bypass"

    # --- Step override ---------------------------------------------------------

    def step(self, action):
        # Save pre-step state so we can detect pickups/drops for shaping.
        prev_carrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)

        # self.step_count += 1
        if action == 2:
            self.forward_action_count += 1
        if action == 3:
            self.items_picked_count += 1

        # --- behavioural counters (continuous-control statistics) -------------
        # Counted every step (including terminal ones). Raw counts are exposed
        # in info and episode_summary; the control formulas normalise them at
        # dataset-load time, so the weights stay tunable without re-collection.
        ax, ay = self.agent_pos
        if isinstance(self.grid.get(ax, ay), HazardTile):
            self.lava_steps += 1
        neighbours = [(ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)]
        if any(
            0 <= nx < self.grid.width and 0 <= ny < self.grid.height
            and isinstance(self.grid.get(nx, ny), HazardTile)
            for nx, ny in neighbours
        ):
            self.lava_adjacent_steps += 1

        living_enemies = [
            pos for alive, pos in (
                (self.enemy1_alive, self.enemy1_pos),
                (self.enemy2_alive, self.enemy2_pos),
            ) if alive and pos is not None
        ]
        min_enemy_d = min(
            (self._manhattan(self.agent_pos, p) for p in living_enemies),
            default=float("inf"),
        )
        if min_enemy_d == 1:
            self.enemy_adjacent_steps += 1
        if min_enemy_d <= 2 and not self._agent_has_camouflage():
            self.enemy_near_unprotected_steps += 1
        if self._is_in_any_detection(self.agent_pos):
            self.detection_cone_steps += 1

        info = dict(info)
        info["lava_steps"] = self.lava_steps
        info["lava_adjacent_steps"] = self.lava_adjacent_steps
        info["enemy_adjacent_steps"] = self.enemy_adjacent_steps
        info["enemy_near_unprotected_steps"] = self.enemy_near_unprotected_steps
        info["detection_cone_steps"] = self.detection_cone_steps

        # --- shaping: bypass step penalty + pickup/drop/door-opened bonuses ----
        # All shaping rewards fire only when target_style matches the action
        # being shaped, so off-target episodes stay sparse.
        if self.shape_rewards:
            if self.target_style == "bypass":
                reward += self.bypass_step_penalty
            elif self.target_style in ("weapon", "camouflage", "daredevil"):
                # per-step time pressure so the agent prefers the *nearest*
                # enemy / zone / lava crossing. Since item side is randomised,
                # the nearest target tracks the item, making routes bimodal
                # (top-item episodes favour the top route, and vice versa).
                reward += self.style_step_penalty

            # pickup
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

            # drop (first drop after first pickup of that item)
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

        # distance-to-nearest-living-enemy metric
        for alive, pos in [
            (self.enemy1_alive, self.enemy1_pos),
            (self.enemy2_alive, self.enemy2_pos),
        ]:
            if alive and pos is not None:
                d = self._manhattan(self.agent_pos, pos)
                self.min_distance_to_enemy = min(self.min_distance_to_enemy, d)
                self.sum_distance_to_enemy += d

        # --- bypass corridor constraint -----------------------------------------
        # When training a route-specific bypass PPO, entering the "wrong" half
        # is a hard failure. This removes the opposite corridor as a winning
        # solution, so the only positive-return behaviour is the chosen route.
        # The layout stays physically identical (both corridors open), so the
        # observations match the true shared env. Start and goal sit on y == 8,
        # so the boundary row is shared by both routes.
        if self.target_style == "bypass" and self.bypass_corridor is not None:
            ay = self.agent_pos[1]
            wrong = (
                (self.bypass_corridor == "upper" and ay > 8)
                or (self.bypass_corridor == "lower" and ay < 8)
            )
            if wrong:
                terminated = True
                reward += self.wrong_corridor_penalty
                info = dict(info)
                info["termination"] = "wrong_corridor"
                info["style"] = None
                info["detected"] = self.detected
                return obs, reward, terminated, truncated, info

        # --- lava check (must happen before detection so deaths are consistent)
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, HazardTile):
            if self._agent_has_boots():
                # first-lava-with-boots shaping (once)
                if self.shape_rewards and not self.first_lava_bonus_paid:
                    self.first_lava_bonus_paid = True
                    if self.target_style == "daredevil":
                        reward += self.first_lava_bonus
                self.traversed_lava_with_boots = True
            elif self.end_on_lava:
                terminated = True
                self.died_in_lava = True
                reward += self.lava_penalty
                info = dict(info)
                info["termination"] = "died_in_lava"
                info["style"] = None
                info["detected"] = self.detected
                return obs, reward, terminated, truncated, info

        # --- detection check ----------------------------------------------------
        if self._is_in_any_detection(self.agent_pos):
            if self._agent_has_camouflage():
                # Camouflage credit only in the (optionally side-restricted)
                # target zone. With a side set, traversing the non-target zone
                # with camo is still safe (no detection death) but earns no
                # camouflage credit, so only the chosen route yields positive
                # return.
                if self._in_camo_target_detection(self.agent_pos):
                    # first-detection-with-camo shaping (once)
                    if self.shape_rewards and not self.first_detection_bonus_paid:
                        self.first_detection_bonus_paid = True
                        if self.target_style == "camouflage":
                            reward += self.first_detection_bonus
                    self.traversed_detection_with_camo = True
            elif self.end_on_detection:
                terminated = True
                self.detected = True
                reward += self.detection_penalty
                info = dict(info)
                info["termination"] = "detected"
                info["style"] = None
                info["detected"] = True
                return obs, reward, terminated, truncated, info

        # --- weapon attack ------------------------------------------------------
        # When a side is set via bypass_corridor, only the target enemy can be
        # killed for credit; toggling next to the non-target enemy does nothing.
        # This biases the run toward one enemy purely through the reward, leaving
        # the layout (both weapons, both enemies) untouched.
        if action == self.actions.toggle and self._agent_has_weapon():
            target = self._weapon_target_enemy()
            killed_this_step = False
            if (target in (None, 1)) and self.enemy1_alive and self._is_adjacent(self.agent_pos, self.enemy1_pos):
                self._kill_enemy(1)
                killed_this_step = True
            elif (target in (None, 2)) and self.enemy2_alive and self._is_adjacent(self.agent_pos, self.enemy2_pos):
                self._kill_enemy(2)
                killed_this_step = True
            if killed_this_step and self.shape_rewards and self.target_style == "weapon":
                reward += self.kill_bonus

        # --- post-kill exit shaping (ordered waypoints) -------------------------
        # After the kill the policy still avoids the now-safe cone/gate region and
        # the col-15 perimeter it never trained on, so it stalls at the gate. Pull
        # it along explicit waypoints: gate -> perimeter corner (col 15) -> goal,
        # advancing as each is reached and re-baselining. Stronger coef than the
        # pre-kill legs. Telescopes per leg, so the optimum is unchanged. Only
        # active for a side-specific weapon run.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "weapon"
            and self.killed_with_weapon
        ):
            gate = self._weapon_exit_gate()
            if gate is not None:
                wps = [gate, (15, gate[1]), self.goal_pos]
                while (
                    self._exit_wp_idx < len(wps) - 1
                    and self._manhattan(self.agent_pos, wps[self._exit_wp_idx]) <= 1
                ):
                    self._exit_wp_idx += 1
                    self._prev_exit_dist = None   # re-baseline on advance
                target = wps[self._exit_wp_idx]
                cur = self._manhattan(self.agent_pos, target)
                if self._prev_exit_dist is not None:
                    reward += 2.0 * self.exit_approach_coef * (self._prev_exit_dist - cur)
                self._prev_exit_dist = cur

        # --- camouflage exit shaping --------------------------------------------
        # Same trap as the weapon case: after collecting the one-time
        # detection-with-camo bonus, the policy stalls in / near the cone and
        # won't push on to the gate (advancing means moving toward the enemy,
        # which it learned to avoid). Once it has entered the target zone, give
        # the same potential-based pull toward the side gate, then the goal.
        # Potential-based, so it telescopes to zero and leaves the optimum
        # unchanged. Only active for a side-specific camouflage run.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "camouflage"
            and self.traversed_detection_with_camo
        ):
            gate = self._weapon_exit_gate()  # side gate: 'lower' -> (14,11)
            if gate is not None:
                in_corridor = self.agent_pos[0] >= gate[0]
                waypoint = self.goal_pos if in_corridor else gate
                phase = "goal" if in_corridor else "gate"
                cur = self._manhattan(self.agent_pos, waypoint)
                if self._exit_phase == phase and self._prev_exit_dist is not None:
                    reward += self.exit_approach_coef * (self._prev_exit_dist - cur)
                self._prev_exit_dist = cur
                self._exit_phase = phase

        # --- bypass perimeter shaping -------------------------------------------
        # Bypass has the longest path and (unlike the other styles) no positive
        # dense signal, so the sparse goal bonus is hard to discover. Give a
        # potential-based pull that walks the agent along the safe outer ring
        # via ordered waypoints, advancing to the next once the current is
        # reached and re-baselining the distance on each advance. Telescopes per
        # leg, so it guides exploration without changing the optimum. Only
        # active for a side-specific bypass run.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "bypass"
            and self.bypass_corridor is not None
        ):
            wps = self._bypass_waypoints()
            if wps is not None:
                while (
                    self._bypass_wp_idx < len(wps) - 1
                    and self._manhattan(self.agent_pos, wps[self._bypass_wp_idx]) <= 1
                ):
                    self._bypass_wp_idx += 1
                    self._prev_bypass_dist = None      # re-baseline on advance
                target = wps[self._bypass_wp_idx]
                cur = self._manhattan(self.agent_pos, target)
                if self._prev_bypass_dist is not None:
                    reward += self.exit_approach_coef * (self._prev_bypass_dist - cur)
                self._prev_bypass_dist = cur

        # --- daredevil boots->lava->goal shaping --------------------------------
        # Unlike weapon/camouflage/bypass, daredevil had no dense pull, so it
        # relied solely on the one-time first_lava_bonus. With a larger view the
        # agent can see the lava strip and learns to avoid it before ever
        # discovering that boots make it safe, getting stuck in a lava-avoidance
        # local optimum. Give a two-phase potential-based pull: first toward the
        # boots, then toward the goal (which sits behind the lava strip, so the
        # pull deliberately routes the now-protected agent across the lava).
        # Potential-based + re-baselined on the phase switch, so it telescopes
        # per leg and leaves the optimum unchanged.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "daredevil"
        ):
            if self._agent_has_boots():
                waypoint = self.goal_pos
                phase = "goal"
            else:
                ground_boots = [
                    b for b in self._boots_positions
                    if isinstance(self.grid.get(*b), Boots)
                ]
                if ground_boots:
                    waypoint = min(
                        ground_boots,
                        key=lambda b: self._manhattan(self.agent_pos, b),
                    )
                    phase = "boots"
                else:
                    waypoint = self.goal_pos
                    phase = "goal"
            cur = self._manhattan(self.agent_pos, waypoint)
            # only reward within a phase; on a phase switch just re-baseline
            if self._dd_phase == phase and self._prev_dd_dist is not None:
                reward += self.exit_approach_coef * (self._prev_dd_dist - cur)
            self._prev_dd_dist = cur
            self._dd_phase = phase

        # --- weapon pre-kill pickup->enemy shaping ------------------------------
        # Weapon only had a *post*-kill pull; pre-kill the sole signal is the
        # sparse goal across the lava on row 8, so the policy heads east into lava.
        # Dense two-phase pull: toward the weapon item, then the target enemy's
        # west (non-cone) attack tile. Post-kill exit shaping then handles
        # gate->goal around the lava.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "weapon"
            and not self.killed_with_weapon
        ):
            waypoint = None
            phase = None
            if not isinstance(self.carrying, Weapon):
                ground = [w for w in self._weapon_positions
                          if isinstance(self.grid.get(*w), Weapon)]
                if ground:
                    waypoint = min(ground, key=lambda w: self._manhattan(self.agent_pos, w))
                    phase = "weapon"
            if waypoint is None:
                tgt = self._weapon_target_enemy()
                cands = []
                if tgt in (None, 1) and self.enemy1_alive:
                    cands.append((self.enemy1_pos[0] - 1, self.enemy1_pos[1]))
                if tgt in (None, 2) and self.enemy2_alive:
                    cands.append((self.enemy2_pos[0] - 1, self.enemy2_pos[1]))
                if cands:
                    waypoint = min(cands, key=lambda p: self._manhattan(self.agent_pos, p))
                    phase = "enemy"
            if waypoint is not None:
                cur = self._manhattan(self.agent_pos, waypoint)
                if self._wpn_phase == phase and self._prev_wpn_dist is not None:
                    reward += self.exit_approach_coef * (self._prev_wpn_dist - cur)
                self._prev_wpn_dist = cur
                self._wpn_phase = phase

        # --- camouflage pre-traversal pickup->cone shaping ----------------------
        # Same trap as weapon: pre-traversal the only pull is the goal across the
        # lava. Dense two-phase pull: toward the camo item, then the side gate
        # (reaching it requires passing through the target detection cone, earning
        # traversal credit while protected). Post-traversal exit shaping then
        # handles gate->goal.
        if (
            self.shape_rewards
            and self.exit_approach_coef
            and self.target_style == "camouflage"
            and not self.traversed_detection_with_camo
        ):
            waypoint = None
            phase = None
            if not isinstance(self.carrying, Camouflage):
                ground = [c for c in self._camo_positions
                          if isinstance(self.grid.get(*c), Camouflage)]
                if ground:
                    waypoint = min(ground, key=lambda c: self._manhattan(self.agent_pos, c))
                    phase = "camo"
            if waypoint is None:
                gate = self._weapon_exit_gate()
                gates = [gate] if gate is not None else [(14, 5), (14, 11)]
                waypoint = min(gates, key=lambda g: self._manhattan(self.agent_pos, g))
                phase = "zone"
            if waypoint is not None:
                cur = self._manhattan(self.agent_pos, waypoint)
                if self._camo_phase == phase and self._prev_camo_dist is not None:
                    reward += self.exit_approach_coef * (self._prev_camo_dist - cur)
                self._prev_camo_dist = cur
                self._camo_phase = phase

        info = dict(info)

        # --- goal check ---------------------------------------------------------
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Goal):
            terminated = True
            achieved = self._achieved_style()
            if self.target_style is None:
                bonus = self.style_bonuses.get(achieved, 0.0)
            else:
                bonus = (
                    self.target_bonus
                    if achieved == self.target_style
                    else self.non_target_penalty
                )

            base = self._reward()
            reward = base + bonus

            info["target_style"] = self.target_style
            info["achieved_style"] = achieved
            info["base_reward"] = base
            info["style_bonus_or_penalty"] = bonus
            info["total_reward"] = reward

            any_enemy_alive = self.enemy1_alive or self.enemy2_alive
            avg_dist = (
                self.sum_distance_to_enemy / max(self.step_count, 1)
                if any_enemy_alive
                else 0.0
            )
            info["episode_summary"] = {
                "total_steps": self.step_count,
                "min_enemy_distance": (
                    self.min_distance_to_enemy
                    if self.min_distance_to_enemy != float("inf")
                    else 0.0
                ),
                "avg_enemy_distance": avg_dist,
                "forward_steps": self.forward_action_count,
                "items_picked": self.items_picked_count,
                "path_efficiency": self.forward_action_count / max(self.step_count, 1),
                "was_detected": self.detected,
                "achieved_style": achieved,
                "killed_with_weapon": self.killed_with_weapon,
                "traversed_lava": self.traversed_lava_with_boots,
                "traversed_detection_with_camo": self.traversed_detection_with_camo,
                "lava_steps": self.lava_steps,
                "lava_adjacent_steps": self.lava_adjacent_steps,
                "enemy_adjacent_steps": self.enemy_adjacent_steps,
                "enemy_near_unprotected_steps": self.enemy_near_unprotected_steps,
                "detection_cone_steps": self.detection_cone_steps,
            }
            return obs, reward, terminated, truncated, info

        info["detected"] = self.detected
        info["enemy1_alive"] = self.enemy1_alive
        info["enemy2_alive"] = self.enemy2_alive
        info["step_count"] = self.step_count
        return obs, reward, terminated, truncated, info


# --- Registration helper -------------------------------------------------------

def register_env():
    gym.envs.registration.register(
        id="MiniGrid-MultiStyles-v0",
        entry_point=MiniGridMultiStyles,
    )


# --- Manual control entry point ------------------------------------------------

if __name__ == "__main__":
    register_env()

    env = gym.make(
        "MiniGrid-MultiStyles-v0",
        target_style="camouflage",
        target_bonus=1.0,
        non_target_penalty=-1.0,
        render_mode="human",
        max_steps=100,
        bypass_corridor=None,
        # show_detection_zones=True,
        free_item_placement=True,
        agent_view_size=7,
    )

    obs, _ = env.reset()
    ret = 0.0
    finish = False

    print("=" * 60)
    print("MiniGrid Multi-Style Env — manual control")
    print("=" * 60)
    print("Controls:")
    print("  LEFT / RIGHT  rotate")
    print("  UP            forward")
    print("  SPACE         pickup")
    print("  TAB           drop")
    print("  DOWN          toggle (use weapon)")
    print("  Z             done")
    print("  R             reset")
    print("=" * 60)

    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True
                break
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_SPACE:
                    action = 3
                elif event.key == pygame.K_TAB:
                    action = 4
                elif event.key == pygame.K_DOWN:
                    action = 5
                elif event.key == pygame.K_z:
                    action = 6
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    ret = 0.0
                    print("\n--- reset ---\n")
                    continue

                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
                    ret += reward
                    print(f"info: {info}")
                    print(f"step reward: {reward:.3f}   episode return: {ret:.3f}")
                    if done or truncated:
                        print(
                            f"\n*** episode end *** "
                            f"(terminated={done}, truncated={truncated}) ***\n"
                            f"final return: {ret:.3f}\n"
                        )
                        finish = True
                        break

        if finish:
            break
        env.render()
        env.unwrapped.clock.tick(10)

    env.close()