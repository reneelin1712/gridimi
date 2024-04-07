import gymnasium as gym
import pygame
from gridEnv import GridEnv

# Instantiate the environment
# env = GridEnv(grid_size=10)
env = gym.make('GridWorld-v0',grid_size=7)

# Number of steps to test
num_steps = 100

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((env.screen_size, env.screen_size))

# Reset the environment at the start of each episode
observation = env.reset()

running = True
step = 0
while running and step < num_steps:
    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Random action for testing
    action = env.action_space.sample()

    # Apply the action to the environment
    observation, reward, done, truncated, info = env.step(action)

    # Render the current state of the environment
    env.render()

    # Update display
    pygame.display.flip()

    # Increment step count
    step += 1

    # Check if the episode is done
    if done:
        print(f"Episode finished after {step} steps")
        break

    # Limit the frame rate
    pygame.time.Clock().tick(10)

# Quit Pygame
pygame.quit()