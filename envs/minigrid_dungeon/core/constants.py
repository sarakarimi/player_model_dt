from __future__ import annotations
import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([231, 76, 60]),
    "dark_red": np.array([194, 54, 22]),
    "green": np.array([46, 204, 113]),
    "blue": np.array([52, 152, 219]),
    "purple": np.array([155, 89, 182]),
    "yellow": np.array([241, 196, 15]),
    "grey": np.array([149, 165, 166]),
    "orange": np.array([230, 126, 34]),
    "dark_blue": np.array([44, 62, 80]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "orange": 6}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "enemy": 11,
    "treasure": 12,
    "weapon": 13,
    "enemy": 14,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of agent state names to integers
AGENT_STATE_TO_IDX = {
    "normal": 0,
    "armed": 1,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]
