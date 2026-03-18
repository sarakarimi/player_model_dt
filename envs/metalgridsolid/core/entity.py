from typing import Any
import pygame
import numpy as np

class Entity:
    def __init__(self, position: tuple[int, int], direction: int | None, color: tuple[int, int, int] | None) -> None:
        """
        Initialize the base entity for the game environment (e.g., agent, enemies, objects).
        
        Args:
            position (tuple): The (row, col) position of the entity on the grid.
            direction (int): The facing direction of the entity (0: up, 1: right, 2: down, 3: left). This is optional for items.
            color (tuple): The RGB color for rendering the entity (optional).
        """
        
        self.init_position = np.array(position)
        self.init_direction = np.array(direction) if direction is not None else None
        self.position = self.init_position
        self.direction = self.init_direction

        self.color = color if color else (255, 255, 255)  # Default to white if no color is provided
    
    def reset(self) -> None:
        self.position = self.init_position
        self.direction = self.init_direction
    
    def render(self, screen: Any, cell_size: int) -> None:
        NotImplemented

    def update(self, env_state: Any) -> None:
        NotImplemented