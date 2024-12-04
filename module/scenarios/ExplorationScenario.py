import random
import time
from abc import ABC

import pygame
import numpy as np

from enums.PointStatus import PointStatus as Point
from enums.DoneStatus import DoneStatus as Done
from logging_system.logger import logging
import const as c
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
        self.max_steps = self.grid_size ** 2 * self.num_agents * 10 # TEST поставить среднее значение для миссии
        self.min_steps = self.grid_size ** 2 * 2
        self.reward_complexion = c.REWARD_DONE * self.count_targets
        self.reward_coef = 1  # TEST динамический коэф

    def _get_scenario_obs(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells. Choose maximum value of status.
        Current agent coordination append to list of all agents positions into his cell.
        :return: Observation dictionary
        """
        agent_obs = self.agents[self.current_agent].get_observation()
        max_coords_status = np.maximum(agent_obs['coords'], self.current_map)
        self.current_map = max_coords_status
        obs = {'pos': [(0, 0) for _ in range(self.num_agents)], 'coords': max_coords_status}
        obs['pos'][self.current_agent] = agent_obs['pos']
        return obs

    def _get_system_reward(self, obs, new_position, agent):
        """
        If cell is not explored, it had status - viewed or empty.
        And in this case cell is not target.
        :param obs: Observation of system
        :param new_position: Coordination of agent
        :param agent: Current agent
        :return: Updated system observation and current agent reward for scenario
        """
        reward = 0
        self.reward_coef *= 1.001 # dynamical coefficient

        x, y = new_position
        if obs['coords'][x][y][0] in (Point.viewed.value, Point.empty.value):
            obs['coords'][x][y][0] = Point.visited.value
        else:
            reward -= c.PENALTY_RETURN
            # reward -= c.PENALTY_RETURN * self.reward_coef
            logging.info(f"{agent.name} тут уже кто-то был {new_position} + {reward}")

        for pos in agent.get_review():
            x, y = pos
            if (x, y) in self.target_positions:
                if obs['coords'][x][y][2] == Done.empty.value:
                    obs['coords'][x][y][2] = Done.done.value
                    reward += c.REWARD_EXPLORE
                    # TEST если делать динамическую награду
                    # reward += c.REWARD_EXPLORE * self.reward_coef
                    logging.info(f"{agent.name} исследовал новую клетку {x, y} + {reward}")
        return obs, reward

    def _check_scenario_termination(self) -> tuple:
        """
        Check conditions for exit game: quantity of steps and if all tasks are done.
        :return: Tuple of conditions (bool, bool, dictionary)
        """
        reward = 0
        terminated = False
        truncated = False
        info = {"done": int(sum(element[2] == Done.done.value for row
                                in self.current_map for element in row)),
                "agent": self.current_agent}

        if self.step_count >= self.max_steps:
            logging.info("Достигнуто максимальное количество шагов в миссии. ")
            truncated = True

        elif info["done"] == self.count_targets:
            terminated = True
            logging.info("Все поле исследовано")
            [setattr(agent, "position", random.choice(self.base_positions)) for agent in self.agents]
            logging.info("Агенты вернулись на базу")
            if self.step_count <= self.min_steps:
                self.total_reward += self.reward_complexion * 1.5
                reward = self.total_reward
                logging.info(f"Увеличенная награда: {reward} за шагов меньше, чем {self.min_steps}")
            else:
                self.total_reward += self.reward_complexion
                reward = self.total_reward
                logging.info(f"Награда: {reward}")
        else:
            self.total_reward += self.step_reward
        #     reward = self.total_reward  # TEST Динамический ревард или 0?
        return reward, terminated, truncated, info

    def _render_scenario(self):
        """Render agent game"""
        cell = self.cell_size
        plant = load_image(c.DONE_TARGET_SPRAY, cell)
        agent_icon = load_image(c.AGENT, cell)

        known_obstacles, known_targets = 0, 0
        for i, flower in enumerate(self.plants_positions):
            x, y = flower
            if self.current_map[x, y, 0] != 0:
                self.screen.blit(plant, (x * cell, y * cell))

        for i, obstacle in enumerate(self.obstacle_positions):
            x, y = obstacle
            if self.current_map[x, y, 0] != 0:
                known_obstacles += 1
                obstacle_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
                self.screen.blit(obstacle_icon, (x * cell, y * cell))

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.current_map[x, y, 0] == 0:
                    dark_overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))
                    self.screen.blit(dark_overlay, (x * cell, y * cell))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(agent_icon, (agent.position[0] * cell,
                                          agent.position[1] * cell))

        # Отрисовка времени, очков, заряда и уровня воды
        screen_width, screen_height = self.screen.get_size()
        status_bar_height = c.BAR_HEIGHT
        elapsed_time = time.time() - self.start_time

        font_size = int(status_bar_height * 0.25)
        font = pygame.font.SysFont(c.FONT, font_size)

        text_x1 = screen_width * 0.05
        text_x2 = screen_width * 0.5
        text_y1 = self.screen_size + status_bar_height * 0.1
        text_y2 = text_y1 + status_bar_height // 4
        text_y3 = text_y1 + status_bar_height // 4 * 2

        color = c.BLACK
        render_text(self.screen, f"Время: {elapsed_time:.2f} сек", font, color, text_x1, text_y1)
        render_text(self.screen, f"Очки: {int(self.total_reward)}", font, color, text_x1, text_y2)
        render_text(self.screen, f"Шагов: {self.step_count}", font, color, text_x1, text_y3)
        render_text(self.screen, f"Обнаружено препятствий: {known_obstacles}/{self.count_obstacles}", font, color,
                    text_x2, text_y1)
        render_text(self.screen,
                    f"Отработано целей:"
                    f" {int(np.sum(element[2] == Done.done.value for row in self.current_map for element in row))}"
                    f"/{self.count_targets}",
                    font, color,
                    text_x2, text_y2)
        pygame.display.flip()
        # time.sleep(0.1) ###

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
        self.obstacle_positions = c.FIXED_OBSTACLE_POSITIONS
        self.target_positions = self._get_objects_positions(
            c.FIXED_TARGET_POSITIONS, self.inner_grid_size * 2 - self.base_size * 2)
