from typing import Tuple
import gymnasium as gym
import pygame
from gymnasium import spaces
from metalgridsolid.core.map import Ground, Vent, Obstacle
import numpy as np
from metalgridsolid.core.enemy import Enemy
from metalgridsolid.core.agent import Agent
from metalgridsolid.core.item import Item
from metalgridsolid.core.constants import DIRECTIONS, Color, EnvID
from metalgridsolid.utils.logger import Logger

class EnvState:
    def __init__(self, 
                agent: Agent,
                enemies: Tuple[Enemy],
                items: Tuple[Item],
                ground: Tuple[Tuple[int, int]],
                vents: Tuple[Tuple[int, int]],
                obstacle: Tuple[Tuple[int, int]],
                map: Tuple[int, int]):
        self.agent = agent
        self.enemies = enemies
        self.items = items
        self.ground = ground
        self.vents = vents
        self.obstacle = obstacle
        self.map = map
        return
    

class MetalGridSolidEnv(gym.Env):
    def __init__(self,
                 agent: Agent,
                 grid_size: Tuple[int, int] = (10, 10),
                 enemies: Tuple[Enemy] | None = None,
                 goal: Tuple[int, int] = (6, 6),
                 walls: Tuple[Tuple[int, int]] | None = None,
                 vents: Tuple[Tuple[Tuple[int, int]]] | None = None,
                 items: Tuple[Item] | None = None,
                 obstacles: Tuple[Tuple[int, int]] = None,
                 rw_weights: dict[str, float] = None,
                 observation_mode: str = 'image',
                 max_steps: int = 100):
        
        super(MetalGridSolidEnv, self).__init__()

        # Initialize pygame
        self.window_size = 600
        self.grid_size = grid_size
        self.width, self.height = grid_size
        self.cell_size = self.window_size // self.width
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Metal Grid Solid")

        # Setup Ground
        self.walls = walls if walls else []
        ground = self._generate_ground()
        assert len(ground) > 0, "No walkable ground found in the grid!"
        self.ground = Ground(position=ground)

        # Setup Vents
        self.vents = [self._generate_vent(vent) for vent in vents] if vents else []
        self.vents = [item for sublist in self.vents for item in sublist]  # Flatten the list of vent positions
        self.vents = Vent(position=self.vents)

        # Setup Obstacles
        self.obstacles = Obstacle(position=obstacles) if obstacles else Obstacle(position=[])

        # Setup Agent
        self.agent = agent

        # Setup Items
        self.items = items

        # Setup Goal
        self.goal = goal

        # Setup Enemies
        self.enemies = enemies
        self.n_enemies = len(self.enemies)
        
        # Setup Reward Weights
        self.rw_weights = {
            'goal': 1.0,            # Standard weight for reaching the goal
            'takedown': 0.0,        # No weight for enemies killed
            'camouflage': 0.0        # No weight for using the item
        } if rw_weights is None else rw_weights

        # To manage game clock
        self.clock = pygame.time.Clock()

        # Setup Gym
        self.max_number_of_steps = max_steps
        self.observation_mode = observation_mode
        
        # Define the observation space based on the mode
        if self.observation_mode == 'features':

            low = np.array([0, -1, -1])  # Object ID: 0 to 6, Direction: -1 to 3, State: -1 to 6
            high = np.array([6, 3, 6])

            # Repeat the low and high arrays to match the grid shape
            low = np.tile(low, (self.grid_size[0], self.grid_size[1], 1))
            high = np.tile(high, (self.grid_size[0], self.grid_size[1], 1))
            
            # Feature-based observation space
            self.observation_space = spaces.Box(
                low=low, 
                high=high,
                shape=(self.height, self.width, 3), 
                dtype=np.int32
            )
        elif self.observation_mode == 'image':
            
            # Image-based observation space
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(self.height * self.cell_size, self.width * self.cell_size, 3), 
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
        
        # Define action space
        self.action_space = spaces.Discrete(6)  # 0: Rotate Left, 1: Rotate Right, 2: Move Forward, 3: Pick Up, 4: Drop Down, 5: Attack

        # Create Environment State
        self.env_state = EnvState(agent=self.agent,
                             enemies=self.enemies,
                             items=self.items,
                             ground=self.ground.position,
                             vents=self.vents.position,
                             obstacle=self.obstacles.position,
                             map=[self.height, self.width])
        
        # Create Logger
        self.logger = Logger()

        # Cache the feature observation for the static-elements
        self.static_ft_obs = self._generate_static_ft_obs()

    
    def reset(self):
        # Reset logger
        self.logger.reset()

        # Reset agent and enemies
        self.agent.reset()
        
        for enemy in self.enemies:
            enemy.reset()

        # Creste Environment State
        self.env_state = EnvState(agent=self.agent,
                             enemies=self.enemies,
                             items=self.items,
                             ground=self.ground.position,
                             vents=self.vents.position,
                             obstacle=self.obstacles.position,
                             map=[self.height, self.width])
        return self._get_obs()
    
    def _generate_static_ft_obs(self):
        # Initialize the observation with empty tiles (OBJECT_IDX = 0, DIRECTION = -1, STATE = -1)
        obs = np.zeros((self.height, self.width, 3), dtype=int)
        obs[:, :, 1] = -1  # Set DIRECTION to -1 for all cells
        obs[:, :, 2] = -1  # Set STATE to -1 for all cells

        # Fill the observation with ground
        for x, y in self.ground.position:
            obs[x, y, 0] = EnvID.GROUND
        
        # Fill the observation with vents
        for x, y in self.vents.position:
            obs[x, y, 0] = EnvID.VENT
        
        # Fill the observation with obstacles
        for x, y in self.obstacles.position:
            obs[x, y, 0] = EnvID.OBSTACLE
        
        return obs

    def _generate_ground(self) -> list[Ground]:
        # Create a grid initialized to False (non-walkable)
        grid = np.full((self.height, self.width), False, dtype=bool)
        
        # Mark cells occupied by walls as True (non-walkable)
        for (row, col) in self.walls:
            grid[row, col] = True

        # Identify bounded regions (areas inside the walls) as walkable
        ground = set()
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if not grid[row, col] and self._is_enclosed(grid, row, col):
                    ground.add((row, col))

        return ground
    
    def _generate_vent(self, vent_points:list[tuple[int, int]]) -> list[tuple[int, int]]:
        # Remove duplicate points while preserving order
        seen = set()
        vent_points = [point for point in vent_points if not (point in seen or seen.add(point))]
        return vent_points

    def _is_enclosed(self, grid, row, col):
        """Check if a cell is enclosed by walls."""
        return (
            any(grid[r, col] for r in range(0, row)) and  # Wall above
            any(grid[r, col] for r in range(row + 1, self.height)) and  # Wall below
            any(grid[row, c] for c in range(0, col)) and  # Wall to the left
            any(grid[row, c] for c in range(col + 1, self.width))  # Wall to the right
        )

    def _find_valid_start_position(self):
        """Finds a valid start position for the agent within the walkable area."""
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in self.ground.position:
                    return np.array([row, col])
        raise ValueError("No valid starting position found in the walkable area!")
    
    def _get_feature_obs(self):

        obs = np.copy(self.static_ft_obs)

        # Fill the observation with the agent' position, direction, and state
        agent_x, agent_y = self.agent.position
        obs[agent_x, agent_y, 0] = EnvID.AGENT
        obs[agent_x, agent_y, 1] = self.agent.direction
        obs[agent_x, agent_y, 2] = self.agent.state

        # Fill the observation with the enemies' position, direction, and state
        for enemy in self.enemies:
            enemy_x, enemy_y = enemy.position
            obs[enemy_x, enemy_y, 0] = EnvID.ENEMY
            obs[enemy_x, enemy_y, 1] = enemy.direction
            obs[enemy_x, enemy_y, 2] = enemy.state
        
        # Fill the observation with the items' position, direction, and state
        for item in self.items:
            item_x, item_y = item.position
            obs[item_x, item_y, 0] = EnvID.CAMOUFLAGE if item.item_type == 'camouflage' else NotImplemented

        return obs
    
    def _get_image_obs(self):
        # Render the current state of the environment
        self.render()  # Ensure the rendering is up to date

        # Capture the image from the screen
        image = pygame.surfarray.array3d(self.screen)
        image = np.transpose(image, (1, 0, 2))  # Convert from (width, height, channels) to (height, width, channels)

        # Convert the numpy array to a PIL image
        # image = Image.fromarray(image)

        # # Resize the image to the specified size (64x64)
        # image = image.resize((84,84), Image.ANTIALIAS)

        # # Save the resized image to a file
        # image.save(f'env_image{self.logger.steps_taken}.png')
        
        return image
    
    def _get_obs(self):
        if self.observation_mode == 'features':
            return self._get_feature_obs()
        elif self.observation_mode == 'image':
            return self._get_image_obs()
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")

    def step(self, action: int):
        
        # Log Step
        self.logger.log_step()

        # Action of the agent
        self.agent.step(action, self.env_state, self.logger)

        # Actions of the enemies
        for enemy in self.enemies:
            enemy.update(self.env_state)

        # Reward function
        reward, done, truncated = self._reward_function()

        return self._get_obs(), reward, done, truncated, {}

    def _reward_function(self) -> tuple[float, bool, bool]:
        
        # Check if we reached maxiumum number of steps
        if self.logger.steps_taken > self.max_number_of_steps:
            return -1.0, False, True

        # Check if the agent is detected by any of the enemies 
        for enemy in self.enemies:
            if enemy.line_of_sight(self.env_state):
                return -1.0, True, False  # Detected by enemy, Lose the game
        
        # Check if the agent has reached the goal - if yes, we need to compute a reward function!
        if self._check_goal():
            # Compute the reward for reaching the goal
            rw_goal = self.rw_weights['goal']*1.0

            # Compute the reward for taking down enemies
            rw_takedown = self.rw_weights['takedown']*self.logger.enemies_killed/self.n_enemies

            # Compute the reward for using the camouflage item
            if "camouflage" in self.logger.items_used and "camouflage" not in self.logger.items_dropped:
                rw_item_used = self.rw_weights['camouflage']*1.0
            else:
                rw_item_used = 0.0

            rw_value = rw_goal + rw_takedown + rw_item_used
            return rw_value, True, False  # Reached the goal, Win the game
        else:
            return 0.0, False, False


    def _check_goal(self):
        return True if np.array_equal(self.agent.position, self.goal) else False


    def render(self, mode="human"):
        self.screen.fill(Color.GREY)

        # Draw ground
        self.ground.render(self.screen, self.cell_size)

        # Draw vents
        self.vents.render(self.screen, self.cell_size)

        # Draw obstacles
        self.obstacles.render(self.screen, self.cell_size)

        # Draw goal
        pygame.draw.rect(self.screen, Color.GREEN, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size))

        # Draw agent
        self.agent.render(self.screen, self.cell_size)

        # Draw enemies
        for enemy in self.enemies:
            enemy.render(self.screen, self.cell_size)

        # Draw items
        for item in self.items:
            item.render(self.screen, self.cell_size)
        
        # Draw grid lines using lines instead of rect borders
        line_color = Color.GREY
        line_thickness = 1  # 1 pixel thickness for thin lines

        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, line_color, (x, 0), (x, self.window_size), line_thickness)
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, line_color, (0, y), (self.window_size, y), line_thickness)
        
        # Draw the border around the full window
        window_rect = pygame.Rect(0, 0, self.window_size, self.window_size)
        pygame.draw.rect(self.screen, Color.WHITE, window_rect, 15)

        pygame.display.flip()

    