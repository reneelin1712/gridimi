import pygame
import numpy as np
import gymnasium as gym

env = gym.make('GridWorld-v0',grid_size=7)
observation,_ = env.reset()

# Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((env.screen_size, env.screen_size))

# Initialize lists to store trajectories and a counter for trajectories
all_states = []
all_actions = []
current_states = []
current_actions = []
trajectory_count = 0
max_trajectories = 10  # Set the maximum number of trajectories to collect

running = True
while running and trajectory_count < max_trajectories:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 2
            elif event.key == pygame.K_DOWN:
                action = 4
            elif event.key == pygame.K_LEFT:
                action = 3
            elif event.key == pygame.K_RIGHT:
                action = 1
            elif event.key == pygame.K_1:
                action = 0
            else:
                continue

            # Record the current state and action
            current_states.append(observation)
            current_actions.append(action)

            # Step through the environment with the chosen action
            observation, reward, done, _, info = env.step(action)

            if action == 0 or done:
                current_states.append(observation)
                # End of a trajectory, save it and start a new one
                all_states.append(current_states)
                all_actions.append(current_actions)
                current_states = []
                current_actions = []
                observation,_ = env.reset()
                trajectory_count += 1  # Increment trajectory count

    # Render the environment
    env.render()

    # Update the display
    pygame.display.flip()

print('all_states',all_states)
# Convert each trajectory to a numpy array and store in a list
states_array = [np.array(trajectory) for trajectory in all_states]
actions_array = [np.array(trajectory) for trajectory in all_actions]

# Save the lists as npy files (note: you will need to load them as lists of arrays)
np.save('../data/states.npy', states_array, allow_pickle=True)
np.save('../data/actions.npy', actions_array, allow_pickle=True)


# Quit Pygame
pygame.quit()
