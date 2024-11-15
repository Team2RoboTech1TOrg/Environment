from abc import ABC

import numpy as np
import pygame

import const
from scenarios.BaseScenario import BaseScenario


class FarmingScenario(BaseScenario, ABC):
    def __init__(self,  num_agents: int,  grid_size: int):
        self.grid_size = grid_size
        self.cell_size = const.SCREEN_SIZE // self.grid_size
        self.margin = const.MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + const.BAR_HEIGHT))
        self.step_reward = None
        self.num_agents = num_agents
        self.target_positions = None
        self.obstacle_positions = None

    def render(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        self.reset_objects_positions()
        self.step_reward = 0

    def reset_objects_positions(self):
        """
        Reset positions of objects
        :return: function for get object's postitions
        """
        if const.PLACEMENT_MODE == 'random':
            self._randomize_positions()
        elif const.PLACEMENT_MODE == 'fixed':
            self._fixed_positions()
        else:
            raise ValueError("Invalid PLACEMENT_MODE. Choose 'random' or 'fixed'.")

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        pass

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        pass

    def _get_available_positions(self, unavailable: set) -> list:
        """
        Function for get available positions from all positions - unavailable
        :param unavailable: set
        :return: available positions
        """
        all_positions = [
            (i, j) for i in range(self.margin, self.inner_grid_size + 1)
            for j in range(self.margin, self.inner_grid_size + 1)
        ]
        return [pos for pos in all_positions if pos not in unavailable]

    def _get_objects_positions(self, unavailable: (), size: int) -> list:
        """
        Get list of object's positions using unavailable positions
        :param unavailable: set()
        :param size: int
        :return: list of positions [x, y]
        """
        available_positions = self._get_available_positions(unavailable)
        indices = np.random.choice(len(available_positions), size=size, replace=False)
        return [available_positions[i] for i in indices]
