import time
import numpy as np
from envs.metalgridsolid.core.item import Camouflage
from envs.metalgridsolid.environment import MetalGridSolidEnv
from envs.metalgridsolid.core.enemy import PatrollingEnemy, StandingEnemy
from envs.metalgridsolid.core.agent import Agent
from envs.metalgridsolid.utils.utils import create_horizontal_limit, create_vertical_limit


if __name__ == "__main__":

    grid_size = (15, 15)

    # Define walls to create a the base level room
    walls = []

    walls += create_horizontal_limit(row=0, col_start=0, col_end=10)
    walls += create_vertical_limit(col=0, row_start=0, row_end=5)
    walls += create_vertical_limit(col=10, row_start=0, row_end=5)
    walls += create_horizontal_limit(row=5, col_start=0, col_end=7)
    walls += create_horizontal_limit(row=5, col_start=9, col_end=10)
    walls += create_vertical_limit(col=7, row_start=5, row_end=5)
    walls += create_vertical_limit(col=6, row_start=6, row_end=13)
    walls += create_horizontal_limit(row=14, col_start=6, col_end=8)
    walls += create_vertical_limit(col=8, row_start=13, row_end=14)
    walls += create_vertical_limit(col=9, row_start=5, row_end=7)
    walls += create_horizontal_limit(row=7, col_start=9, col_end=11)
    walls += create_vertical_limit(col=11, row_start=7, row_end=11)
    walls += create_horizontal_limit(row=11, col_start=9, col_end=11)
    walls += create_vertical_limit(col=9, row_start=11, row_end=13)


    # Define vent paths
    vent_l = []
    vent_l += create_horizontal_limit(row=6, col_start=3, col_end=6)
    vent_l += create_vertical_limit(col=3, row_start=6, row_end=12)
    vent_l += create_horizontal_limit(row=12, col_start=3, col_end=6)

    vent_r = []
    vent_r += create_horizontal_limit(row=6, col_start=9, col_end=12)
    vent_r += create_vertical_limit(col=12, row_start=6, row_end=12)
    vent_r += create_horizontal_limit(row=12, col_start=9, col_end=12)


    # Define agent
    agent = Agent(position=(1, 1), direction=2)

    # Define enemies
    enemies = [PatrollingEnemy(position=(3, 1), direction=1),
               StandingEnemy(position=(10, 9), direction=3)]
    
    # Define goal position
    goal = (13, 7)

    # Define obstacles
    obstacles = [(9, 9)]

    # Define items
    items = [Camouflage(position=(1, 9))]

    # Reward weigths
    rw_weights = {
        'goal': 1.0,            # Standard weight for reaching the goal
        'takedown': 0.0,        # No weight for enemies killed
        'camouflage': 0.0       # No weight for using the camouflage
}   
    # Define max steps of the environment
    max_steps = 100

    # Setup visualization
    observation_mode = 'features' # Image or features
    render_mode = 'array' # Array or human (for rendering)


    # Create the environment
    env = MetalGridSolidEnv(agent=agent,
                            enemies=enemies,
                            vents=[vent_l, vent_r],
                            items=items,
                            grid_size=grid_size,
                            walls=walls,
                            obstacles=obstacles,
                            goal=goal,
                            rw_weights=rw_weights,
                            observation_mode=observation_mode,
                            render_mode=render_mode,
                            max_steps=max_steps)
    obs, _ = env.reset()
    done = False

    # Benchmarking loop
    steps = 0
    total_reward = 0

    start_time = time.time()
    while not steps >= max_steps:
        action = np.random.choice(env.action_space.n)  # Choose a random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if done or truncated:
            env.reset()
    end_time = time.time()

    # Calculate steps per second
    elapsed_time = end_time - start_time
    steps_per_second = steps / elapsed_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds.")
    print(f"Steps per second: {steps_per_second:.2f} steps/second.")