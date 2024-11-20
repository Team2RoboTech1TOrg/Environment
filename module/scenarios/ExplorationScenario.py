import random
import time
from abc import ABC

import pygame
import numpy as np

from PointStatus import PointStatus, ObjectStatus
from logger import logging
import const
from scenarios.FarmingScenario import FarmingScenario
from utils import load_image


class ExplorationScenario(FarmingScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        super().__init__(num_agents, grid_size)
        self.start_time = None
        self.name = 'exploration'
        self.target_positions = None
        self.obstacle_positions = None

    def _reset_scenario(self, *, seed=None, options=None):
        self.start_time = time.time()

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
        return obs, 0

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

        elif np.all(self.done_status == 1):
            terminated = True
            logging.info("Все растения опрысканы")
            for agent in self.agents:
                agent.position = random.choice(self.base_positions)
            logging.info("Агенты вернулись на базу")
            # logging.info(self.current_map)

            # условие по времени выполнения
            if self.step_count <= const.MIN_GAME_STEPS:
                total_reward = self.total_reward + const.REWARD_COMPLETION * 3
                logging.info(f"Увеличенная награда: {total_reward}за шагов меньше, чем {const.MIN_GAME_STEPS}")
            else:
                total_reward = self.total_reward + const.REWARD_COMPLETION
                logging.info(f"Награда: {total_reward}")
            # self.total_reward = 0
        else:
            self.total_reward += self.step_reward
            total_reward = 0  # self.total_reward VS 0 проверить как лучше будет работать
        return total_reward, terminated, truncated, {}

    def _render_scenario(self):
        """Render agent game"""
        target_done_icon = load_image(const.DONE_TARGET_EXPLORE, self.cell_size)
        agent_icon = load_image(const.AGENT, self.cell_size)

        known_obstacles, known_targets = 0, 0
        for i, target in enumerate(self.target_positions):
            x, y = target
            if self.current_map[x, y, 0] != 0:
                known_targets += 1
                if self.done_status[i]:
                    self.screen.blit(target_done_icon, (y * self.cell_size, x * self.cell_size))

        for i, obstacle in enumerate(self.obstacle_positions):
            x, y = obstacle
            if self.current_map[x, y, 0] != 0:
                known_obstacles += 1
                obstacle_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
                self.screen.blit(obstacle_icon, (y * self.cell_size, x * self.cell_size))

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.current_map[x, y, 0] == 0:
                    dark_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))  # Непрозрачный
                    self.screen.blit(dark_overlay, (y * self.cell_size, x * self.cell_size))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(agent_icon, (agent.position[1] * self.cell_size,
                                          agent.position[0] * self.cell_size))

        # Отрисовка времени, очков, заряда и уровня воды
        screen_width, screen_height = self.screen.get_size()
        status_bar_height = const.BAR_HEIGHT
        elapsed_time = time.time() - self.start_time  # Рассчитываем время

        font_size = int(status_bar_height * 0.25)  # Размер шрифта для панели статуса
        font = pygame.font.SysFont('Arial', font_size)

        text_x1 = screen_width * 0.05
        text_x2 = screen_width * 0.5
        text_y1 = const.SCREEN_SIZE + status_bar_height * 0.1
        text_y2 = text_y1 + status_bar_height // 4
        text_y3 = text_y1 + status_bar_height // 4 * 2

        self.screen.blit(font.render(f"Время: {elapsed_time:.2f} сек", True, const.BLACK),
                         (text_x1, text_y1))
        self.screen.blit(font.render(f"Очки: {int(self.total_reward)}", True, const.BLACK),
                         (text_x1, text_y2))
        self.screen.blit(font.render(f"Шаги: {self.step_count}", True, const.BLACK),
                         (text_x1, text_y3))

        self.screen.blit(font.render(f"Обнаружено препятствий: {known_obstacles}/{const.COUNT_OBSTACLES}",
                                     True, const.BLACK), (text_x2, text_y1))
        self.screen.blit(
            font.render(f"Обнаружено целей: {known_targets}/{len(self.target_positions)}",
                        True, const.BLACK), (text_x2, text_y2))
        self.screen.blit(font.render(f"Отработано целей: {int(np.sum(self.done_status))}/"
                                     f"{len(self.target_positions)}", True, const.BLACK),
                         (text_x2, text_y3))

        pygame.display.flip()

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        unavailable_positions = set(self.base_positions)
        # TO DO позиции вокруг базы по 1 клетке нельзя препятствия + это учесть при запросе размера поля

        # TO DO  проверка, чтоб они не стояли вокруг пустой клетки
        self.obstacle_positions = self._get_objects_positions(unavailable_positions, const.COUNT_OBSTACLES)
        unavailable_positions.update(self.obstacle_positions)

        self.target_positions = self._get_available_positions(unavailable_positions)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.obstacle_positions = const.FIXED_OBSTACLE_POSITIONS
        self.target_positions = self._get_objects_positions(
            const.FIXED_TARGET_POSITIONS, self.inner_grid_size * 2 - const.STATION_SIZE * 2)
