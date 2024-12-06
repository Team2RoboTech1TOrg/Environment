import logging
from collections import deque

import numpy as np
import pygame

from abc import ABC, abstractmethod
from math import ceil

from gymnasium.core import ActType

import const as c
from agent.Agent import Agent
from enums.ObjectStatus import ObjectStatus as Obj
from scenarios.BaseScenario import BaseScenario
from utils import load_obstacles, load_image


class FarmingScenario(BaseScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        self.name = None
        self.screen_size = c.SCREEN_SIZE
        self.grid_size = grid_size
        self.cell_size = self.screen_size // self.grid_size
        self.margin = c.MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        self.screen = None
        self.num_agents = num_agents
        self.count_targets = None
        self.target_positions = None
        self.count_obstacles = ceil(self.grid_size ** 2 * c.OBSTACLE_PERCENT)
        self.obstacle_positions = None
        self.count_plants = ceil(self.grid_size ** 2 * c.TARGET_PERCENT)
        self.plants_positions = None
        self.base_size = c.STATION_SIZE
        base_coords = (self.grid_size // 2 - self.base_size // 2, self.margin + 1)
        self.base_positions = [(base_coords[0] + i, base_coords[1] + j) for i in range(self.base_size)
                               for j in range(self.base_size)]
        self.agents = [Agent(self, name=f'agent_{i}') for i in range(self.num_agents)]
        self.obstacle_icons = load_obstacles(c.OBSTACLES, self.cell_size, self.count_obstacles)
        self.current_map = None
        self.reward_coef = None
        self.total_reward = None
        self.step_count = None
        self.max_steps = None
        self.min_steps = None
        self.step_reward = None
        self.reward_complexion = None
        self.current_agent = None
        self.agents_positions = None
        self.all_terminated = None
        self.all_truncated = None

    def reset(self, *, seed=None, options=None):
        self.reset_objects_positions()
        self.step_count = 1
        self.total_reward = 0
        self.all_terminated = deque(maxlen=self.num_agents)
        self.all_truncated = deque(maxlen=self.num_agents)
        self.current_map = np.full((self.grid_size, self.grid_size, 3), fill_value=0)
        for x, y in self.base_positions:
            self.current_map[x][y][1] = Obj.base.value
        agent_obs = [agent.reset() for agent in self.agents]
        self.agents_positions = [obs['pos'] for obs in agent_obs]
        obs = {'pos': self.agents_positions,
               'coords': np.max(np.stack([obs['coords'] for obs in agent_obs]), axis=0)}
        self.current_agent = 0
        self._reset_scenario()
        return obs, {}

    @abstractmethod
    def _reset_scenario(self):
        pass

    def get_observation(self):
        obs = self._get_scenario_obs()
        return obs

    @abstractmethod
    def _get_scenario_obs(self):
        pass

    def step(self, action: ActType):
        logging.info(f"Шаг: {self.step_count}")
        obs = self.get_observation()
        self.step_reward = 0
        idx = self.current_agent
        agent = self.agents[idx]

        new_position, agent_reward, agent_terminated, agent_truncated, info = agent.take_action(action)
        self.all_terminated.append(agent_terminated)
        self.all_truncated.append(agent_truncated)
        self.step_reward += agent_reward

        if self._check_system_termination():
            # self.step_reward = 0
            # self.total_reward = 0
            return obs, self.step_reward, agent_terminated, agent_truncated, info

        obs, system_reward = self._get_system_reward(obs, new_position, agent)
        obs['pos'][idx] = new_position
        self.step_reward += system_reward

        termination_reward, terminated, truncated, info = self._check_scenario_termination()
        self.step_reward += termination_reward

        self.current_map = np.maximum(obs['coords'], self.current_map)
        self.total_reward += self.step_reward
        self.step_count += 1
        logging.info(
            f"Награда: {ceil(self.total_reward)}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )
        self.current_agent = (idx + 1) % self.num_agents
        return obs, self.step_reward, terminated, truncated, info

    def _check_system_termination(self) -> bool:
        """
        Check are all agents terminated or truncated mission.
        :return: True or false
        """
        if all(self.all_terminated):
            return True
        elif all(self.all_truncated):
            self.step_reward -= 300
            print(self.step_reward)
            return True

    @abstractmethod
    def _get_system_reward(self, obs: dict[str, np.array], new_position: tuple[int, int], agent: Agent):
        pass

    @abstractmethod
    def _check_scenario_termination(self):
        pass

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + c.BAR_HEIGHT))
        cell = self.cell_size
        bg = pygame.image.load(c.FIELD_BACKGROUND).convert()
        base_icon = load_image(c.STATION, cell)
        bg_image = pygame.image.load(c.FIELD).convert()

        full_field_size = self.grid_size * cell
        bg = pygame.transform.smoothscale(bg, (full_field_size, full_field_size))

        # Отрисовка сетки
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(
                    self.screen, c.BLACK,
                    (x * cell, y * cell, cell, cell), 1
                )
        # Отрисовка границы внутреннего поля
        inner_field_size = self.inner_grid_size * self.cell_size
        margin_x = (self.grid_size * cell - inner_field_size) // 2
        margin_y = (self.grid_size * cell - inner_field_size) // 2
        inner_field_rect = pygame.Rect(margin_x, margin_y, inner_field_size, inner_field_size)
        pygame.draw.rect(self.screen, c.BLACK, inner_field_rect, 4)

        self.screen.blit(bg, (0, 0))
        bg_image = pygame.transform.smoothscale(bg_image, (inner_field_size, inner_field_size))
        self.screen.blit(bg_image, (margin_x, margin_y))

        # Отрисовка панели статуса
        status_bar_rect = pygame.Rect(0, self.screen_size, self.screen_size,
                                      c.BAR_HEIGHT)
        pygame.draw.rect(self.screen, c.WHITE, status_bar_rect)

        # Отрисовка базы
        base_size = self.base_size * self.cell_size
        base_icon_scaled = pygame.transform.smoothscale(base_icon, (base_size, base_size))
        x, y = self.base_positions[0]
        self.screen.blit(base_icon_scaled,
                         (x * cell, y * cell))
        screen_width, screen_height = self.screen.get_size()
        status_bar_height = c.BAR_HEIGHT

        font_size = int(status_bar_height * 0.25)
        font = pygame.font.SysFont(c.FONT, font_size)

        text_x1 = int(screen_width * 0.05)
        text_x2 = int(screen_width * 0.5)
        text_y1 = self.screen_size + status_bar_height * 0.1
        text_y2 = text_y1 + status_bar_height // 4
        text_y3 = text_y1 + status_bar_height // 4 * 2
        self._render_scenario(font, text_x1, text_x2, text_y1, text_y2, text_y3)

    @abstractmethod
    def _render_scenario(self, font: pygame.font, text_x1: int, text_x2: int, text_y1: int, text_y2: int,
                         text_y3: int):
        pass

    def render_message(self, render_text: str):
        """
        Display message in the center of screen
        :param render_text: str
        :return:
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size + c.BAR_HEIGHT))
        self.screen.fill(c.GRAY)

        font = pygame.font.Font(pygame.font.get_default_font(), int(self.screen_size * 0.055))
        lines = render_text.split('\n')
        screen_width, screen_height = self.screen.get_size()
        y_offset = (screen_height - len(lines) * font.get_height()) // 2

        for i, line in enumerate(lines):
            if i == 0:
                font_title = pygame.font.Font(pygame.font.get_default_font(), ceil(font.get_height() * 1.2))
                text_surface = font_title.render(line, True, c.RED)
                text_width, text_height = font_title.size(line)
            else:
                text_surface = font.render(line, True, c.GREEN)
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
        if c.PLACEMENT_MODE == 'random':
            self._randomize_positions()
        elif c.PLACEMENT_MODE == 'fixed':
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
        Function for get available positions from all positions - unavailable.
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
        Get list of object's positions using unavailable positions.
        :param unavailable: set()
        :param size: int
        :return: list of positions [x, y]
        """
        available_positions = self._get_available_positions(unavailable)
        indices = np.random.choice(len(available_positions), size=size, replace=False)
        return [available_positions[i] for i in indices]

    def get_restricted_area_around_base(self) -> set:
        """
        Get all positions around base.
        """
        restricted_positions = {
            (x + dx, y + dy) for x, y in self.base_positions
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size
        }
        return restricted_positions

    def __repr__(self):
        return f'scenario_{self.name}'
