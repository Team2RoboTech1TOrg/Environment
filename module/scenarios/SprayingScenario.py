import random
import time
from abc import ABC

import pygame
import numpy as np

from enums.PointStatus import PointStatus as Point
from enums.DoneStatus import DoneStatus as Done
from enums.ObjectStatus import ObjectStatus as Obj
from logging_system.logger import logging
import const as c
from render.menu_render import render_text
from scenarios.FarmingScenario import FarmingScenario
from utils import load_image


class SprayingScenario(FarmingScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        super().__init__(num_agents, grid_size)
        self.start_time = None
        self.name = 'spraying'
        self.count_targets = self.count_plants

    def _reset_scenario(self, *, seed=None, options=None):
        self.start_time = time.time()
        self.max_steps = self.grid_size ** 2 * self.num_agents * 10  # TEST поставить среднее значение для миссии
        self.min_steps = self.grid_size ** 2 * self.num_agents
        self.reward_complexion = c.REWARD_DONE * self.count_targets
        self.reward_coef = 1

    def _get_scenario_obs(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells. Choose maximum value of status.
        Current agent coordination append to list of all agents positions into his cell.
        :return: Observation dictionary
        """
        idx = self.current_agent
        agent_obs = self.agents[idx].get_observation()
        max_coords_status = np.maximum(agent_obs['coords'], self.current_map)
        self.current_map = max_coords_status
        self.agents_positions[idx] = agent_obs['pos']
        obs = {'pos': self.agents_positions, 'coords': max_coords_status}
        return obs

    def _get_system_reward(self, obs, new_position, agent):
        """
        If cell is not explored, agent get reward for explore it.
        Also, if cell is target, agent get reward.
        If this cell has status visited, agent get penalty.
        And in this case cell is not target.
        :param obs: Observation of system
        :param new_position: Coordination of agent
        :param agent: Current agent
        :return: Updated system observation and current agent reward for scenario
        """
        reward = 0
        self.reward_coef *= 1.001  # dynamical coefficient
        x, y = new_position

        if obs['coords'][x][y][0] == Point.visited.value:
            reward -= c.PENALTY_RETURN
            # reward -= c.PENALTY_RETURN * self.reward_coef
            logging.info(f"{agent.name} тут уже кто-то был {new_position} + {reward}")
        elif obs['coords'][x][y][0] == Point.viewed.value:
            obs['coords'][x][y][0] = Point.visited.value
            reward = c.REWARD_EXPLORE
            # reward = const.REWARD_EXPLORE * self.reward_coef
            logging.info(f"{agent.name} исследовал новую клетку {new_position} + {reward}")

        # Можно сделать штраф только если там растение и кто-то был
        if obs['coords'][x][y][1] == Obj.plant.value and obs['coords'][x][y][2] == Done.empty.value:
            agent.energy -= c.ENERGY_CONSUMPTION_DONE
            agent.tank -= c.ON_TARGET_CONSUMPTION
            obs['coords'][x][y][2] = Done.done.value
            reward += c.REWARD_DONE
            # reward = const.REWARD_EXPLORE * self.reward_coef
            logging.info(f"{self} выполнена задача {new_position}, награда {reward, 2}")
        return obs, reward

    def _check_scenario_termination(self) -> tuple:
        """
        Check conditions for exit game: quantity of steps and if all targets are done.
        :return: tuple of conditions (bool, bool, dictionary)
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
            self.step_reward -= self.reward_complexion * 0.5

        elif info["done"] == self.count_targets:
            terminated = True
            logging.info("Все поле исследовано")
            [setattr(agent, "position", random.choice(self.base_positions)) for agent in self.agents]
            logging.info("Агенты вернулись на базу")
            if self.step_count <= self.min_steps:
                reward = self.reward_complexion * 2
                logging.info(f"Увеличенная награда: {reward} за шагов меньше, чем {self.min_steps}")
            else:
                reward = self.reward_complexion
                logging.info(f"Награда за выполненную миссию: {reward}")
        return reward, terminated, truncated, info

    def _render_scenario(self, font: pygame.font, text_x1: int, text_x2: int, text_y1: int, text_y2: int,
                         text_y3: int):
        """Render agent game"""
        cell = self.cell_size
        target_icon = load_image(c.TARGET_SPRAY, cell)
        target_done_icon = load_image(c.DONE_TARGET_SPRAY, cell)
        agent_icon = load_image(c.AGENT, cell)

        known_targets, known_obstacles = 0, 0
        for i, target in enumerate(self.target_positions):
            x, y = target
            if self.current_map[x, y, 0] != Point.empty.value:
                known_targets += 1
            icon = target_done_icon if self.current_map[x, y, 2] == 1 else target_icon
            self.screen.blit(icon, (x * cell, y * cell))

        for i, obstacle in enumerate(self.obstacle_positions):
            x, y = obstacle
            if self.current_map[x, y, 0] != Point.empty.value:
                known_obstacles += 1
            obstacle_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
            self.screen.blit(obstacle_icon, (x * cell, y * cell))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.current_map[x, y, 0] == Point.empty.value:
                    dark_overlay = pygame.Surface((cell, cell), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 100))
                    self.screen.blit(dark_overlay, (x * cell, y * cell))

        for agent in self.agents:
            self.screen.blit(agent_icon, (agent.position[0] * cell,
                                          agent.position[1] * cell))

        elapsed_time = time.time() - self.start_time

        color = c.BLACK
        render_text(self.screen, f"Время: {elapsed_time:.2f} сек", font, color, text_x1, text_y1)
        render_text(self.screen, f"Очки: {int(self.total_reward)}", font, color, text_x1, text_y2)
        render_text(self.screen, f"Шагов: {self.step_count}", font, color, text_x1, text_y3)
        render_text(self.screen, f"Обнаружено препятствий: {known_obstacles}/{self.count_obstacles}", font, color,
                    text_x2, text_y1)
        render_text(self.screen, f"Обнаружено целей: {known_targets}/{self.count_targets}", font, color,
                    text_x2, text_y2)
        render_text(self.screen,
                    f"Отработано целей:"
                    f" {int(np.sum(element[2] == Done.done.value for row in self.current_map for element in row))}"
                    f"/{self.count_targets}",
                    font, color,
                    text_x2, text_y3)
        pygame.display.flip()

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        unavailable_positions = self.get_restricted_area_around_base()
        self.target_positions = self._get_objects_positions(unavailable_positions, self.count_targets)
        unavailable_positions.update(self.target_positions)
        self.obstacle_positions = self._get_objects_positions(unavailable_positions, self.count_obstacles)

        while any(self._is_surrounded_by_obstacles(target) for target in self.target_positions):
            self.obstacle_positions = self._get_objects_positions(unavailable_positions, self.count_obstacles)

        self.plants_positions = self.target_positions

    def _is_surrounded_by_obstacles(self, target_position):
        """
        Check if a flower is surrounded by obstacles on 3 sides.
        :param target_position: (x, y) position of the flower
        :return: True if the flower is surrounded by obstacles on 3 sides, False otherwise
        """
        x, y = target_position
        step = 1
        surrounding_positions = [
            (x - step, y), (x + step, y), (x, y - step), (x, y + step)
        ]
        obstacle_count = sum(pos in self.obstacle_positions for pos in surrounding_positions)
        return obstacle_count == 3

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = c.FIXED_TARGET_POSITIONS
        self.obstacle_positions = c.FIXED_OBSTACLE_POSITIONS
