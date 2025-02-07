from dataclasses import dataclass
from typing import Tuple

class Color:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0) # Ground
    RED = (237, 60, 57) # Agent
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0) # Goal
    GREY = (99, 99, 99)  # Outside world
    BLUE = (52, 152, 219) # Patroling Enemies
    ORANGE = (230, 126, 34) # Standing Enemies
    BROWN = (193, 154, 108) # Camouflage
    DARK_BROWN = (101, 67, 33) # Obstacles
    LIGHT_GREY = (189, 195, 199)  # Vents

# Direction vectors for [UP, RIGHT, DOWN, LEFT]
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

@dataclass(frozen=True)
class AgentAction:
    NOOP: int = 0
    LEFT: int = 1
    RIGHT: int = 2
    FORWARD: int = 3
    PICK_UP: int = 4
    DROP_DOWN: int = 5
    ATTACK: int = 6

@dataclass(frozen=True)
class EnemyState:
    PATROLLING: int = 0
    STANDING: int = 1
    SENTRY: int = 2
    CHASING: int = 3
    RANDOM: int = 4
    INCAPACITATED: int = 5
    DEAD: int = 6

@dataclass(frozen=True)
class AgentState:
    HIDDEN: int = 0
    VISIBLE: int = 1
    CAMOUFLAGED: int = 2

@dataclass(frozen=True)
# Map of object type to integers
class EnvID:
    EMPTY: int = 0
    GROUND: int = 1
    VENT: int = 2
    OBSTACLE: int = 3
    ENEMY: int = 4
    AGENT: int = 5
    CAMOUFLAGE: int = 6