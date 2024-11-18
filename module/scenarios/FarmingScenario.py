import numpy as np
import pygame

from abc import ABC
from math import ceil

import const
from Agent import Agent
from scenarios.BaseScenario import BaseScenario


class FarmingScenario(BaseScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        self.start_time = None
        self.grid_size = grid_size
        self.cell_size = const.SCREEN_SIZE // self.grid_size
        self.margin = const.MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        self.screen = None  # Экран создается при необходимости
        self.num_agents = num_agents
        base_coords = (self.margin + 1, self.grid_size // 2 - const.STATION_SIZE // 2)
        self.base_positions = [(base_coords[0] + i, base_coords[1] + j) for i in range(const.STATION_SIZE)
                               for j in range(const.STATION_SIZE)]
        self.agents = [Agent(self, name=f'agent_{i}') for i in range(self.num_agents)]

    def reset(self):
        pass
        # self.reset_objects_positions()

    def step(self, action):
        pass

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + const.BAR_HEIGHT))

    def render_message(self, render_text: str):
        """
        Display message in the center of screen
        :param render_text: str
        :return:
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + const.BAR_HEIGHT))
        self.screen.fill(const.GRAY)

        font = pygame.font.Font(pygame.font.get_default_font(), int(const.SCREEN_SIZE * 0.055))
        lines = render_text.split('\n')
        screen_width, screen_height = self.screen.get_size()
        y_offset = (screen_height - len(lines) * font.get_height()) // 2

        for i, line in enumerate(lines):
            if i == 0:
                font_title = pygame.font.Font(pygame.font.get_default_font(), ceil(font.get_height() * 1.2))
                text_surface = font_title.render(line, True, const.RED)
                text_width, text_height = font_title.size(line)
            else:
                text_surface = font.render(line, True, const.GREEN)
                text_width, text_height = font.size(line)
            x_offset = (screen_width - text_width) // 2
            self.screen.blit(text_surface, (x_offset, y_offset))
            y_offset += text_height + 5
        pygame.display.flip()

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
