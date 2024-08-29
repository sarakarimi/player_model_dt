from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from envs.minigrid_dungeon.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from envs.minigrid_dungeon.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle
)

if TYPE_CHECKING:
    from envs.minigrid_dungeon.minigrid_env import MiniGridEnv

Point = Tuple[int, int]


class WorldObj:

    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return False
    
    def can_attack(self) -> bool:
        """Can the agent atack this?"""
        return False

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return False

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "treasure":
            v = Treasure()
        elif obj_type == "weapon":
            v = Weapon()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__("door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        # if door is closed and unlocked
        elif not self.is_open:
            state = 1
        else:
            raise ValueError(
                f"There is no possible state encoding for the state:\n -Door Open: {self.is_open}\n -Door Closed: {not self.is_open}\n -Door Locked: {self.is_locked}"
            )

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True
    

# class Treasure(WorldObj):
#     def __init__(self, color="yellow"):
#         super().__init__("treasure", color)  # 'ball' as placeholder type

#     def can_pickup(self):
#         return True

#     def render(self, img):
#         c = COLORS[self.color]  # Make sure this color is defined in your COLORS dictionary

#         def create_star_mask(dimensions, num_points=5, inner_ratio=0.5):
#             """
#             Create a mask for a star shape based on normalized coordinates (from 0 to 1),
#             with vertical alignment and centered perfectly.
#             """
#             # Center of the star in normalized coordinates
#             cx, cy = 0.5, 0.5
#             radius = 0.5  # Outer radius (half of normalized width)
#             points = []

#             # Calculate the starting angle to align the first outer point directly upwards
#             starting_angle = -np.pi / 2  # Starting from the top (-90 degrees)

#             # Generate points for outer and inner vertices
#             for i in range(num_points * 2):
#                 angle_rad = starting_angle + i * 2 * np.pi / (num_points * 2)
                
#                 if i % 2 == 0:
#                     # Outer point
#                     x = cx + radius * np.cos(angle_rad)
#                     y = cy + radius * np.sin(angle_rad)
#                 else:
#                     # Inner point
#                     x = cx + inner_ratio * radius * np.cos(angle_rad)
#                     y = cy + inner_ratio * radius * np.sin(angle_rad)
                
#                 points.append((x, y))

#             def star_fn(x, y):
#                 """ Function to determine if a point is inside the star, based on normalized coordinates. """
#                 count = 0
#                 for i in range(len(points)):
#                     x0, y0 = points[i]
#                     x1, y1 = points[(i + 1) % len(points)]
#                     # Check if the point is in the half-plane towards the interior of the polygon
#                     if ((y0 <= y < y1) or (y1 <= y < y0)) and \
#                         (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
#                         count += 1

#                 return count % 2 == 1  # Inside if crossed an odd number of polygon sides

#             return star_fn
            
#         mask_fn = create_star_mask(img.shape)
#         fill_coords(img, mask_fn, c)


class Treasure(WorldObj):
    def __init__(self, color="yellow"):
        super().__init__("treasure", color)  # 'ball' as placeholder type

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]  # Make sure this color is defined in your COLORS dictionary

        def create_coins_mask(dimensions):
            """
            Create a mask for a star shape based on normalized coordinates (from 0 to 1),
            with vertical alignment and centered perfectly.
            """
            coins = [
                (0.25, 0.7, 0.2),  # coin 1 at center (0.3, 0.5) with radius 0.05
                (0.5, 0.3, 0.2),  # coin 2
                (0.75, 0.7, 0.2)   # coin 3
            ]

            def coins_fn(x, y):
                for x_center, y_center, radius in coins:
                    if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                        return True
                return False

            return coins_fn
            
        mask_fn = create_coins_mask(img.shape)
        fill_coords(img, mask_fn, c)


class Weapon(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("weapon", color)  # 'ball' as placeholder type

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]  # Make sure this color is defined in your COLORS dictionary

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

    

class Enemy(WorldObj):
    def __init__(self, color="purple"):
        super().__init__("enemy", color)  # 'ball' as placeholder type

    def can_attack(self):
        return True
    
    def render(self, img):
        c = COLORS[self.color]  # Make sure this color is defined in your COLORS dictionary

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