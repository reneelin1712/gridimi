import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import pygame
import sys


class GridEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=7):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(5)  # 0: stop, 1: right, 2: up, 3: left, 4: down
        # Observations are array with the agent's location. Each location is encoded as a number
        self.observation_space = spaces.Box(low=-self.grid_size, high=(self.grid_size + 1) * (self.grid_size + 1) - 1,
                                            shape=(1,), dtype=np.int64)
        self.agent_location = np.array([0, 0])

        # Pygame setup
        self.cell_size = 50
        self.screen_size = self.grid_size * self.cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()


    def step(self, action):
        self.step_count += 1
        done = False

        if action == 0:
            done = True
        elif action == 1:  # right
            self.agent_location[0] += 1
        elif action == 2:  # up
            self.agent_location[1] += 1
        elif action == 3:  # left
            self.agent_location[0] -= 1
        elif action == 4:  # down
            self.agent_location[1] -= 1


        matched = np.array_equal(self.agent_location, np.array([self.grid_size - 1, self.grid_size - 1]))
        
        # Check if the agent is outside the grid
        outside_grid = (self.agent_location[0] < 0 or self.agent_location[0] >= self.grid_size) or \
                   (self.agent_location[1] < 0 or self.agent_location[1] >= self.grid_size)
        
        # Assuming grid_size is always >= 2, max distance will be 2*(grid_size-1)
        max_distance = 2 * (self.grid_size - 1)

        if matched:
            done = True 
            reward = 1
        elif self.step_count >= max_distance:
            done = True
            reward = 0
        elif outside_grid:
            done = True
            reward = -1
        else:
            # Negative reward for each step
            reward = -0.2

            # Calculate the Manhattan distance to the target
            distance_to_target = abs(self.agent_location[0] - (self.grid_size - 1)) + \
                                abs(self.agent_location[1] - (self.grid_size - 1))
            
            reward += 0.5 * (1 - (distance_to_target / max_distance))  # Reward based on closeness to target

        truncated = False
        info = {"action_taken": action}
        # print('action',action)
        # print('state',self._get_obs())
        # print('outside_grid',outside_grid)
        # print('done',done)
        return self._get_obs(), reward, done,truncated, info

    def _get_obs(self):
        observation_value = self.grid_size * self.agent_location[1] + self.agent_location[0]
        # print('self.agent_location[1] ',self.agent_location[1] )
        # print('self.agent_location[0]',self.agent_location[0])
        return np.array([observation_value])

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.agent_location = np.array([0, 0])
        return self._get_obs(),{} #observation, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))  # Fill the screen with white

        # Draw grid
        for x in range(0, self.screen_size, self.cell_size):
            for y in range(0, self.screen_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw the agent as a green circle
        agent_x = self.agent_location[0] * self.cell_size
        agent_y = self.agent_location[1] * self.cell_size
        pygame.draw.circle(self.screen, (0, 255, 0), (agent_x, agent_y), self.cell_size // 4)

        # Draw the target as a blue circle
        target_x = (self.grid_size - 1) * self.cell_size
        target_y = (self.grid_size - 1) * self.cell_size
        pygame.draw.circle(self.screen, (0, 0, 255), (target_x, target_y), self.cell_size // 4)

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.display.quit()
        pygame.quit()