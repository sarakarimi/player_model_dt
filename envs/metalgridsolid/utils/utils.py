from typing import Any
import numpy as np
import pygame

def create_horizontal_limit(row, col_start, col_end):
    return [(row, col) for col in range(col_start, col_end+1)]

def create_vertical_limit(col, row_start, row_end):
    return [(row, col) for row in range(row_start, row_end+1)]

def validate_position(position, existing_positions, walkable_area):
    if position not in walkable_area:
        raise ValueError(f"Position {position} is not within the walkable area!")
    if position in existing_positions:
        raise ValueError(f"Position {position} overlaps with another element!")
    return True

def flatten_obs(obs):
    flat_obs = []
    def _extract_values(d):
        for key, value in d.items():
            if isinstance(value, dict):
                _extract_values(value)
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    for item in value:
                        _extract_values(item)
                else:
                    flat_obs.extend(np.array(value).flatten().tolist())
            elif isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten().tolist())
            else:
                flat_obs.append(float(value))
    _extract_values(obs)
    return flat_obs


def check_valid_position(position: tuple[int, int], env_state: Any, agent_check: bool = False) -> bool:
     
    # Check if the position is within the bounds of the map
    if not (0 <= position[0] < env_state.map[0] and 0 <= position[1] < env_state.map[1]): # Height, Width
        return False
     
    # Check if position is in the ground area
    if agent_check:
        if tuple(position) not in env_state.ground and tuple(position) not in env_state.vents:
            return False
    else:
        if tuple(position) not in env_state.ground:
            return False
     
     # Check if position is not blocked by obstacles
    if tuple(position) in env_state.obstacle:
        return False
     
     # Check if position is blocked by other entities
    if np.array_equal(position, env_state.agent.position):
        return False
    
    for enemy in env_state.enemies:
        if np.array_equal(position, enemy.position):
            return False
    
    for item in env_state.items:
        if np.array_equal(position, item.position) and item.is_picked_up == False:
            return False
    return True


def draw_triangle(position, direction, color, img, cell_size, scale=0.6):
    # Calculate the center of the cell
    x, y = position[1] * cell_size, position[0] * cell_size
    cx, cy = x + cell_size // 2, y + cell_size // 2

    # Calculate the size of the triangle based on the scale
    size = int((cell_size // 2) * scale)  # Scale down the size of the triangle

    # Define the points of the triangle based on the direction
    if direction == 0:  # UP
        points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    elif direction == 1:  # RIGHT
        points = [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]
    elif direction == 2:  # DOWN
        points = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
    elif direction == 3:  # LEFT
        points = [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]
    
    # Draw the triangle manually in the NumPy array
    fill_polygon(img, points, np.array(color, dtype=np.uint8))


def fill_polygon(img, points, color):
    # Extract the bounding box of the polygon
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    
    # Iterate over the bounding box and fill pixels inside the triangle
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_triangle((x, y), points):
                img[y, x] = color

def point_in_triangle(p, triangle):
    """Check if point p is inside the triangle defined by the list of points."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(p, triangle[0], triangle[1]) < 0.0
    b2 = sign(p, triangle[1], triangle[2]) < 0.0
    b3 = sign(p, triangle[2], triangle[0]) < 0.0

    return ((b1 == b2) and (b2 == b3))