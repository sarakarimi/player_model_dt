import numpy as np
from metalgridsolid.core.entity import Entity
from metalgridsolid.core.constants import Color

class Obstacle():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.DARK_BROWN
    
    def render(self, img, cell_size) -> None:
        for pos in self.position:
            start_y = pos[0] * cell_size
            start_x = pos[1] * cell_size
            end_y = start_y + cell_size
            end_x = start_x + cell_size
            
            # Fill the corresponding area in the image array with the obstacle's color
            img[start_y:end_y, start_x:end_x] = self.color


class Ground():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.BLACK
    
    def render(self, img, cell_size) -> None:
        for pos in self.position:
            start_y = pos[0] * cell_size
            start_x = pos[1] * cell_size
            end_y = start_y + cell_size
            end_x = start_x + cell_size
            
            # Fill the corresponding area in the image array with the ground's color
            img[start_y:end_y, start_x:end_x] = self.color


class Vent():
    def __init__(self, position: tuple[tuple[int, int]]):
        self.position = position
        self.color = Color.LIGHT_GREY
    
    def render(self, img, cell_size) -> None:
        for pos in self.position:
            start_y = pos[0] * cell_size
            start_x = pos[1] * cell_size
            end_y = start_y + cell_size
            end_x = start_x + cell_size
            
            # Fill the corresponding area in the image array with the vent's color
            img[start_y:end_y, start_x:end_x] = self.color


class Goal():
    def __init__(self, position: tuple[int, int]):
        self.position = position
        self.color = np.array(Color.GREEN, dtype=np.uint8)
    
    def render(self, img, cell_size) -> None:
        start_y = self.position[0] * cell_size
        start_x = self.position[1] * cell_size
        end_y = start_y + cell_size
        end_x = start_x + cell_size
        
        # Fill the corresponding area in the image array with the vent's color
        img[start_y:end_y, start_x:end_x] = self.color