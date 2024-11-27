import random
import time
from abc import ABC

import pygame
import numpy as np

from PointStatus import DoneStatus
from logging_system.logger import logging
import const
from render.menu_render import render_text
from scenarios.FarmingScenario import FarmingScenario
from utils import load_image


class ExplorationScenario(FarmingScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        super().__init__(num_agents, grid_size)
        self.start_time = None
        self.name = 'exploration'
        self.count_targets = self.inner_grid_size ** 2

    def _reset_scenario(self, *, seed=None, options=None):
        self.start_time = time.time()
        self.max_steps = self.grid_size ** 2 * 5 # TEST поставить среднее значение для миссии
        self.reward_coef = 1  # new

    def _get_scenario_obs(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells
        """
        agent_obs = [agent.get_observation() for agent in self.agents]
        max_agent_coords = np.max(np.stack([obs['coords'] for obs in agent_obs]), axis=0)
        max_coords_status = np.maximum(max_agent_coords, self.current_map)
        self.current_map = max_coords_status

        obs = {'pos': np.stack([obs['pos'] for obs in agent_obs]),
               'coords': max_coords_status}
        return obs

    def _get_system_reward(self, obs, new_position, agent):
        """
        If cell is not explored, it had status - viewed or empty.
        And in this case cell is not target.
        :param obs:
        :param new_position:
        :param agent:
        :return: observation full and reward for scenario
        """
        reward = 0
        self.reward_coef *= 0.99  # new

        # что видит агент в данной позиции
        for pos in agent.get_review():
            x, y = pos
            if (x, y) in self.target_positions:
                if obs['coords'][x][y][2] == DoneStatus.empty.value:
                    obs['coords'][x][y][2] = DoneStatus.done.value
                    reward = const.REWARD_EXPLORE
                    # TEST если делать динамическую награду
                    # reward = const.REWARD_EXPLORE * self.reward_coef
                    logging.info(f"{agent.name} исследовал новую клетку {x, y} + {round(reward, 2)}")
        return obs, reward

    def _check_termination_conditions(self) -> tuple:
        """
        Check conditions for exit game: quantity of steps and if all flowers are watered.
        :return: tuple of conditions (bool, bool, dictionary)
        """
        terminated = False
        truncated = False
        if self.step_count >= const.MAX_STEPS_GAME:
            logging.info("Достигнуто максимальное количество шагов в миссии. ")
            total_reward = 0
            truncated = True

        elif sum(element[2] == DoneStatus.done.value for row in self.current_map for element in
                 row) == self.count_targets:
            terminated = True
            logging.info("Все растения опрысканы")
            [setattr(agent, 'position', random.choice(self.base_positions)) for agent in self.agents]
            logging.info("Агенты вернулись на базу")

            # условие по времени выполнения
            if self.step_count <= const.MIN_GAME_STEPS:
                self.total_reward += const.REWARD_COMPLETION * 1.2
                total_reward = self.total_reward
                logging.info(f"Увеличенная награда: {total_reward}за шагов меньше, чем {const.MIN_GAME_STEPS}")
            else:
                self.total_reward += const.REWARD_COMPLETION
                total_reward = self.total_reward
                logging.info(f"Награда: {total_reward}")
            # self.total_reward = 0
        else:
            self.total_reward += self.step_reward
            total_reward = self.total_reward  # TEST Динамический ревард или 0? проверить как лучше будет работать
        return total_reward, terminated, truncated, {}

    def _render_scenario(self):
        """Render agent game"""
        cell = self.cell_size
        plant = load_image(const.DONE_TARGET_SPRAY, cell)
        agent_icon = load_image(const.AGENT, cell)

        known_obstacles, known_targets = 0, 0
        for i, flower in enumerate(self.plants_positions):
            x, y = flower
            if self.current_map[x, y, 0] != 0:
                self.screen.blit(plant, (y * cell, x * cell))

        for i, obstacle in enumerate(self.obstacle_positions):
            x, y = obstacle
            if self.current_map[x, y, 0] != 0:
                known_obstacles += 1
                obstacle_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
                self.screen.blit(obstacle_icon, (y * cell, x * cell))

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.current_map[x, y, 0] == 0:
                    dark_overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))
                    self.screen.blit(dark_overlay, (y * cell, x * cell))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(agent_icon, (agent.position[1] * cell,
                                          agent.position[0] * cell))

        # Отрисовка времени, очков, заряда и уровня воды
        screen_width, screen_height = self.screen.get_size()
        status_bar_height = const.BAR_HEIGHT
        elapsed_time = time.time() - self.start_time

        font_size = int(status_bar_height * 0.25)
        font = pygame.font.SysFont(const.FONT, font_size)

        text_x1 = screen_width * 0.05
        text_x2 = screen_width * 0.5
        text_y1 = self.screen_size + status_bar_height * 0.1
        text_y2 = text_y1 + status_bar_height // 4
        text_y3 = text_y1 + status_bar_height // 4 * 2

        color = const.BLACK
        render_text(self.screen, f"Время: {elapsed_time:.2f} сек", font, color, text_x1, text_y1)
        render_text(self.screen, f"Очки: {int(self.total_reward)}", font, color, text_x1, text_y2)
        render_text(self.screen, f"Шагов: {self.step_count}", font, color, text_x1, text_y3)
        render_text(self.screen, f"Обнаружено препятствий: {known_obstacles}/{self.count_obstacles}", font, color,
                    text_x2, text_y1)
        render_text(self.screen,
                    f"Отработано целей:"
                    f" {int(np.sum(element[2] == DoneStatus.done.value for row in self.current_map for element in row))}"
                    f"/{self.count_targets}",
                    font, color,
                    text_x2, text_y2)
        pygame.display.flip()

    def _randomize_positions(self):
        """
        Get random positions of objects.
        """
        self.target_positions = self._get_available_positions(set())
        unavailable_positions = self.get_restricted_area_around_base()

        self.obstacle_positions = self._get_objects_positions(unavailable_positions, self.count_obstacles)
        unavailable_positions.update(self.obstacle_positions)

        self.plants_positions = self._get_objects_positions(unavailable_positions, self.count_plants)

    def _fixed_positions(self):
        """
        Get fixed positions of objects.
        """
        self.obstacle_positions = const.FIXED_OBSTACLE_POSITIONS
        self.target_positions = self._get_objects_positions(
            const.FIXED_TARGET_POSITIONS, self.inner_grid_size * 2 - self.base_size * 2)
