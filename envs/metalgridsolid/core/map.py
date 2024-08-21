import pygame
from metalgridsolid.core.entity import Entity
from metalgridsolid.core.constants import Color


class Obstacle():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.DARK_BROWN
    
    def render(self, screen, cell_size) -> None:
        for pos in self.position:
            pygame.draw.rect(screen, self.color, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))


class Ground():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.BLACK
    
    def render(self, screen, cell_size) -> None:
        for pos in self.position:
            pygame.draw.rect(screen, self.color, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))


class Vent():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.LIGHT_GREY
    
    def render(self, screen, cell_size) -> None:
        for pos in self.position:
            pygame.draw.rect(screen, self.color, (pos[1] * cell_size, pos[0] * cell_size, cell_size, cell_size))