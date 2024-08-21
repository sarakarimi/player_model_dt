import pygame
from typing import Any
from metalgridsolid.core.constants import Color, AgentState
from metalgridsolid.core.entity import Entity


class Item(Entity):
    def __init__(self, position: tuple[int, int], color: tuple[int, int, int], item_type: str) -> None:
        """
        Initialize the item entity for the game environment.
        
        Args:
            position (tuple): The (row, col) position of the item on the grid.
            color (tuple): The RGB color for rendering the item.
            item_type (str): The type of the item (e.g., key, door, etc.).
        """
        
        super().__init__(position, None, color)
        self.item_type = item_type
        self.is_picked_up = False
    
    def reset(self) -> None:
        self.position = self.init_position
        self.is_picked_up = False
    
    def render(self, screen: Any, cell_size: int) -> None:
        NotImplemented

    def pickup(self) -> AgentState:
        NotImplemented

    def drop(self, position) -> AgentState:
        NotImplemented


class Camouflage(Item):
    def __init__(self, position: tuple[int, int]) -> None:
        """
        Initialize the camouflage item entity for the game environment.
        
        Args:
            position (tuple): The (row, col) position of the item on the grid.
        """
        
        super().__init__(position, Color.BROWN, "camouflage")
    
    def pickup(self) -> AgentState:
        self.is_picked_up = True
        return AgentState.CAMOUFLAGED
    
    def drop(self, position) -> AgentState:
        self.is_picked_up = False
        self.position = position
        return AgentState.HIDDEN
    
    def render(self, screen: Any, cell_size: int) -> None:
        if not self.is_picked_up:
            pygame.draw.rect(screen, self.color, (self.position[1] * cell_size, self.position[0] * cell_size, cell_size, cell_size))