from typing import Any
import numpy as np
from metalgridsolid.core.entity import Entity
from metalgridsolid.core.constants import EnemyState, Color, DIRECTIONS, AgentState
from metalgridsolid.utils.utils import check_valid_position, draw_triangle


class Enemy(Entity):
    def __init__(self, position: tuple[int, int], direction: int, init_state: int = EnemyState.STANDING):
        self.state = init_state
        self.init_state = init_state
        self.color = None
        self.set_color()
        super().__init__(position, direction, self.color)

    def reset(self) -> None:
        self.position = self.init_position
        self.direction = self.init_direction
        self.state = self.init_state

    def set_color(self) -> None:
        if self.state == EnemyState.STANDING:
            self.color = Color.ORANGE
        elif self.state == EnemyState.PATROLLING:
            self.color = Color.BLUE
        else:
            NotImplementedError
    
    def set_state(self, state: int) -> None:
        self.state = state
        if state == EnemyState.DEAD:
            self.position = (-1, -1)
            self.direction = -1

    def line_of_sight(self, env_state: Any) -> bool:
        # If Agent is Dead then return False
        if self.state == EnemyState.DEAD:
            return False

        # Get the current state of the environment
        agent_position = env_state.agent.position
        env_ground = env_state.ground
        env_obstacle = env_state.obstacle
        
        # Calculate the position in front of the enemy
        front_position = self.position + DIRECTIONS[self.direction]
        
        # If the agent is in the position directly in front of the enemy, it detects it regardless of the state
        if np.array_equal(agent_position, front_position):
            return True
        
        # If the enemy is camouflaged, it cannot detect the agent
        if env_state.agent.state == AgentState.CAMOUFLAGED:
            return False
        
        if self.direction == 0: # UP
            if self.position[1] == agent_position[1] and self.position[0] > agent_position[0]:
                if all(tuple([x, self.position[1]]) in env_ground and tuple([x, self.position[1]]) not in env_obstacle
                        for x in range(agent_position[0] + 1, self.position[0])):
                        return True
        elif self.direction == 1:  # Enemy facing RIGHT
                if self.position[0] == agent_position[0] and self.position[1] < agent_position[1]:
                    if all(tuple([self.position[0], y]) in env_ground and tuple([self.position[0], y]) not in env_obstacle
                        for y in range(self.position[1] + 1, agent_position[1])):
                        return True
        elif self.direction == 2:  # Enemy facing DOWN
                if self.position[1] == agent_position[1] and self.position[0] < agent_position[0]:
                    if all(tuple([x, self.position[1]]) in env_ground and tuple([x, self.position[1]]) not in env_obstacle
                        for x in range(self.position[0] + 1, agent_position[0])):
                        return True
        elif self.direction == 3:  # Enemy facing LEFT
                if self.position[0] == agent_position[0] and self.position[1] > agent_position[1]:
                    if all(tuple([self.position[0], y]) in env_ground and tuple([self.position[0], y]) not in env_obstacle
                        for y in range(agent_position[1] + 1, self.position[1])):
                        return True

        return False
    
    def render(self, img: Any, cell_size: int) -> None:
        if self.state != EnemyState.DEAD:
            draw_triangle(self.position, self.direction, self.color, img, cell_size)
    

class PatrollingEnemy(Enemy):
    def __init__(self, position: tuple[int, int], direction: int):
        super().__init__(position, direction, init_state=EnemyState.PATROLLING)

    def move(self, env_state: Any) -> None:
        new_position = self.position + DIRECTIONS[self.direction]
            
        # Check if the new position is valid
        if not check_valid_position(new_position, env_state):
            # Reverse direction if the movement is blocked or if the agent or camouflage is in the new position
            self.direction = (self.direction + 2) % 4
            new_position = self.position + DIRECTIONS[self.direction]

        # Ensure the new position is still valid after reversing direction
        if check_valid_position(new_position, env_state):
            self.position = new_position

    def update(self, env_state: Any) -> None:
        self.move(env_state)
        
    
class StandingEnemy(Enemy):
    def __init__(self, position: tuple[int, int], direction: int) -> None:
        super().__init__(position, direction, init_state=EnemyState.STANDING)

    def update(self, env_state: Any) -> None:
        return