from typing import Any
import numpy as np

from metalgridsolid.core.entity import Entity
from metalgridsolid.core.constants import Color, DIRECTIONS, AgentState, AgentAction, EnemyState
from metalgridsolid.utils.utils import check_valid_position, draw_triangle
from metalgridsolid.utils.logger import Logger

class Agent(Entity):
    def __init__(self, position: tuple[int, int], direction: int, init_state: int = AgentState.HIDDEN):
        self.state = init_state
        self.init_state = init_state
        self.color = None
        self.item = None
        self.set_color()
        super().__init__(position, direction, self.color)

    def set_color(self) -> None:
        if self.state == AgentState.HIDDEN:
            self.color = Color.RED
        elif self.state == AgentState.CAMOUFLAGED:
            self.color = Color.BROWN
        else:
            NotImplementedError

    def reset(self) -> None:
        self.position = self.init_position
        self.direction = self.init_direction
        self.state = self.init_state
        self.item = None

    def render(self, screen: Any, cell_size: int) -> None:
        draw_triangle(self.position, self.direction, self.color, screen, cell_size, scale=0.6) 

    def step(self, action: int, env_state: Any, logger: Logger) -> None:
        # Rotate the agent
        if action == AgentAction.NOOP:  # NOOP
            return
        
        elif action == AgentAction.LEFT:  # Rotate Left
            self.direction = (self.direction - 1) % 4
        
        elif action == AgentAction.RIGHT:  # Rotate Right
            self.direction = (self.direction + 1) % 4
        
        elif action == AgentAction.FORWARD:  # Move Forward
            new_position = self.position + DIRECTIONS[self.direction]
            if check_valid_position(new_position, env_state, agent_check=True):
                self.position = new_position

        elif action == AgentAction.PICK_UP:  # Pick Up
            pickup_position = self.position + DIRECTIONS[self.direction]
            for item in env_state.items:
                if np.array_equal(item.position, pickup_position) and item.is_picked_up == False:
                    logger.log_item_used(item.item_type)
                    self.state = item.pickup()
                    self.set_color()
                    self.item = item
                    break

        elif action == AgentAction.DROP_DOWN:  # Drop Down
            if self.item is not None:
                drop_position = self.position + DIRECTIONS[self.direction]
                if check_valid_position(drop_position, env_state, agent_check=True):
                    logger.log_item_dropped(self.item.item_type)
                    for item in env_state.items:
                        if np.array_equal(item.item_type, self.item.item_type) and item.is_picked_up == True:
                            self.state = item.drop(drop_position)
                            self.set_color()
                

        elif action == AgentAction.ATTACK:  # Attack
            # Calculate the position in front of the agent (where it is facing)
            attack_position = self.position + DIRECTIONS[self.direction]
            
            # Check if there is an enemy at the attack_position and if the agent is behind the enemy
            for enemy in env_state.enemies:
                enemy_position, enemy_direction = enemy.position, enemy.direction
                if not np.array_equal(enemy_position, attack_position):
                    continue
                # Calculate the position behind the enemy (relative to the enemy's direction)
                enemy_back_position = enemy_position - DIRECTIONS[enemy_direction]
                
                # Check if the agent is at the enemy's back and facing the enemy
                if np.array_equal(self.position, enemy_back_position) and np.array_equal(attack_position, enemy_position):
                    enemy.set_state(EnemyState.DEAD)
                    logger.log_enemy_killed()
                    break