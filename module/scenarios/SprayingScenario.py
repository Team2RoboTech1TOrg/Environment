import time
from abc import ABC
import random # для костыля keno
import pygame
import gymnasium as gym
import numpy as np

from Agent import Agent
from PointStatus import PointStatus, ObjectStatus
from SystemObservationSpace import SystemObservationSpace
from logger import logging
import const
from scenarios.FarmingScenario import FarmingScenario
from utils import convert_to_multidiscrete, load_image, load_obstacles


class SprayingScenario(FarmingScenario, ABC):
    def __init__(self, num_agents: int, grid_size: int):
        super().__init__(num_agents, grid_size)
        self.obstacle_icons = None
        self.total_reward = None
        self.step_reward = None
        self.done_status = None
        self.target_positions = None
        self.obstacle_positions = None
        self.known_obstacles = None
        self.known_targets = None
        self.viewed_cells = None
        self.current_map = None
        self.step_count = None
        # Рандомная база 2х2
        self.base_size = 2  
        self.base_position = self._get_random_base_position()
        self.base_cells = [(self.base_position[0] + i, self.base_position[1] + j) for i in range(self.base_size) for j in range(self.base_size)]
        self.agent_start_times = [i * 2 for i in range(num_agents)]

    def reset(self, *, seed=None, options=None):
        self.reset_objects_positions()
        self.done_status = np.zeros(const.COUNT_TARGETS)
        self.start_time = time.time()
        self.total_reward = 0
        self.step_reward = 0
        self.step_count = 1
        self.current_map = np.full((self.grid_size, self.grid_size, 3), fill_value=0)
        self.obstacle_icons = load_obstacles(const.OBSTACLES, self.cell_size, const.COUNT_OBSTACLES)

        # Инициализация объектов на карте
        for pos in self.obstacle_positions:
            self.current_map[pos[0], pos[1], 1] = ObjectStatus.obstacle.value

        for idx, pos in enumerate(self.target_positions):
            self.current_map[pos[0], pos[1], 1] = ObjectStatus.target.value
            self.current_map[pos[0], pos[1], 2] = self.done_status[idx]

        agent_obs = [agent.reset() for agent in self.agents]
        obs = {'pos': np.stack([obs['pos'] for obs in agent_obs]),
               'coords': self.current_map.copy()}
        logging.info("Перезагрузка среды")
        return obs, {}

    def get_observation(self):
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

  
    def step(self, actions):
        logging.info(f"Шаг: {self.step_count}")
        self.step_reward = 0

        # Получаем наблюдения от агентов и обновляем общую карту
        agent_obs = [agent.get_observation() for agent in self.agents]
        # Объединяем информацию от всех агентов
        for obs in agent_obs:
            observed_indices = np.where(obs['coords'][:, :, 0] != 0)
            self.current_map[observed_indices] = obs['coords'][observed_indices]
        # max_agent_coords = np.max(np.stack([obs['coords'] for obs in agent_obs]), axis=0)
        # self.current_map = np.maximum(self.current_map, max_agent_coords)

        obs = {'pos': np.stack([obs['pos'] for obs in agent_obs]),
               'coords': self.current_map.copy()}

        for i, agent in enumerate(self.agents):
            if self.step_count >= self.agent_start_times[i]:
                action = actions[i]
                new_position, agent_reward, terminated, truncated, info = agent.take_action(action)
            else:
                logging.info(f"{agent.name} ожидает вылета")
                continue
            if self.step_count != 1:
                new_position = self.check_crash(obs, agent, new_position)
            value_position = obs['coords'][new_position[0]][new_position[1]][0]
            if value_position == PointStatus.empty.value or value_position == PointStatus.viewed.value:
                self.step_reward += const.REWARD_EXPLORE
                logging.info(f"{agent.name} исследовал новую клетку {new_position}")
                self.current_map[new_position[0]][new_position[1]][0] = PointStatus.visited.value
            obs['pos'][i] = new_position
            self.step_reward += agent_reward

        # Обновляем списки известных объектов
        self.known_targets = np.argwhere((self.current_map[:, :, 1] == ObjectStatus.target.value) &
                                         (self.current_map[:, :, 2] == 0))
        self.known_obstacles = np.argwhere(self.current_map[:, :, 1] == ObjectStatus.obstacle.value)
        self.viewed_cells = np.argwhere(self.current_map[:, :, 0] != 0)

        reward, terminated, truncated, info = self._check_termination_conditions()
        self.step_count += 1
        logging.info(
            f"Награда: {self.total_reward}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )

        return obs, reward, terminated, truncated, {}

    def check_crash(self, obs: dict, agent: Agent, new_position: tuple[int, int]):
        """
        Check if agents coordinates is same with another agents.
        :param new_position: position of agent (x, y)
        :param agent: agent in process
        :param obs: all agents positions at the moment
        :return: agent coordinates x, y
        """
        # TO DO проверить работает ли верно = вывод конечной позиции (agent.position)
        for i, item in enumerate(obs['pos']):
            if i != int(agent.name.split('_')[1]) and tuple(item) == new_position:
                self.total_reward -= const.PENALTY_CRASH
                logging.warning(f"Столкнование {new_position} агентов")
                # print((f"Столкнование {new_position} агентов"))
                new_position = agent.position
                logging.info(f"{agent.name} возвращается на предыдущую позицию {new_position}")
        return new_position

    def _check_termination_conditions(self) -> tuple:
        """
        Check conditions for exit game: quantity of steps and if all flowers are watered.
        :return: tuple of conditions (bool, bool, dictionary)
        """
        terminated = False
        truncated = False

        if self.step_count >= const.MAX_STEPS_GAME:  # костыль выхода, потом убрать
            logging.info("Достигнуто максимальное количество шагов")
            total_reward = 0
            truncated = True

        elif self.total_reward <= -500:  # Завершение при достижении награды -50 keno
            logging.info("Достигнут лимит отрицательной награды (-50)")
            terminated = True
            total_reward = self.total_reward
            self.total_reward = 0                                                                                                            
        elif np.all(self.done_status == 1):
            terminated = True
            logging.info("Все растения опрысканы")
            for agent in self.agents:
                # TO DO позиции рандомно из базы
                agent.position = self.base_position
            logging.info("Агенты вернулись на базу")

            # условие по времени выполнения BREAK
            if self.step_count <= const.MIN_GAME_STEPS:
                total_reward = self.total_reward + const.REWARD_COMPLETION * 3
            else:
                total_reward = self.total_reward + const.REWARD_COMPLETION
            logging.info(f"Награда: {total_reward}")
            self.total_reward = 0
        else:
            self.total_reward += self.step_reward
            total_reward = self.step_reward
        return total_reward, terminated, truncated, {}

    def render(self):
        """Render agent game"""
        target_icon = load_image(const.TARGET, self.cell_size)
        target_done_icon = load_image(const.DONE_TARGET, self.cell_size)
        base_icon = load_image(const.STATION, self.cell_size)
        agent_icon = load_image(const.AGENT, self.cell_size)

        bg_image = pygame.image.load(const.FIELD).convert()
        bg = pygame.image.load(const.FIELD_BACKGROUND).convert()

        full_field_size = self.grid_size * self.cell_size
        bg = pygame.transform.smoothscale(bg, (full_field_size, full_field_size))

        # Отрисовка сетки
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(
                    self.screen, const.BLACK,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1
                )

        # Отрисовка границы внутреннего поля (устанавливаем цвет и толщину линии)
        inner_field_size = self.inner_grid_size * self.cell_size
        margin_x = (self.grid_size * self.cell_size - inner_field_size) // 2
        margin_y = (self.grid_size * self.cell_size - inner_field_size) // 2
        inner_field_rect = pygame.Rect(margin_x, margin_y, inner_field_size, inner_field_size)
        pygame.draw.rect(self.screen, const.BLACK, inner_field_rect, 4)

        # Фон общий и фон поля
        self.screen.blit(bg, (0, 0))
        bg_image = pygame.transform.smoothscale(bg_image, (inner_field_size, inner_field_size))
        self.screen.blit(bg_image, (margin_x, margin_y))

       # Отрисовка базы на 2x2 клетках
        for cell in self.base_cells:
            self.screen.blit(base_icon, (cell[1] * self.cell_size, cell[0] * self.cell_size))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(agent_icon, (agent.position[1] * self.cell_size,
                                          agent.position[0] * self.cell_size))
        # # Рисуем цветы и ямы
        # for i, pos in enumerate(self.target_positions):
            # if pos in self.known_targets:
                # if self.done_status[i]:
                    # icon = target_done_icon
                # else:
                    # icon = target_icon
                # self.screen.blit(icon, (pos[1] * self.cell_size, pos[0] * self.cell_size))

        # for i, obstacle in enumerate(self.obstacle_positions):
            # if obstacle in self.known_obstacles:
                # barrier_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
                # self.screen.blit(barrier_icon, (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size))
        # Отрисовка цветков
        for idx, pos in enumerate(self.target_positions):
            x, y = pos
            if self.current_map[x, y, 0] != 0:  # Если клетка была видна хотя бы одному агенту
                if self.current_map[x, y, 2] == 1:
                    icon = target_done_icon  # Цветок полит
                else:
                    icon = target_icon  # Цветок не полит
                self.screen.blit(icon, (y * self.cell_size, x * self.cell_size))

        # Отрисовка препятствий
        for i, pos in enumerate(self.obstacle_positions):
            x, y = pos
            if self.current_map[x, y, 0] != 0:  # Если клетка была видна хотя бы одному агенту
                barrier_icon = self.obstacle_icons[i % len(self.obstacle_icons)]
                self.screen.blit(barrier_icon, (y * self.cell_size, x * self.cell_size))        

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if self.current_map[x, y, 0] == 0:  # Если клетка скрыта
                    dark_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))  # Затемнение
                    self.screen.blit(dark_overlay, (y * self.cell_size, x * self.cell_size))            

        # Отрисовка панели статуса
        screen_width, screen_height = self.screen.get_size()
        status_bar_height = const.BAR_HEIGHT
        status_bar_rect = pygame.Rect(0, const.SCREEN_SIZE, const.SCREEN_SIZE,
                                      status_bar_height)
        pygame.draw.rect(self.screen, const.WHITE, status_bar_rect)

        # Отрисовка времени, очков, заряда и уровня воды
        elapsed_time = time.time() - self.start_time  # Рассчитываем время

        font_size = int(status_bar_height * 0.25)  # Размер шрифта для панели статуса
        font = pygame.font.SysFont('Arial', font_size)

        text_x1 = screen_width * 0.05
        text_x2 = screen_width * 0.5
        text_y1 = const.SCREEN_SIZE + status_bar_height * 0.1
        text_y2 = const.SCREEN_SIZE + status_bar_height // 4 + status_bar_height * 0.1
        text_y3 = const.SCREEN_SIZE + status_bar_height // 4 * 2 + status_bar_height * 0.1

        self.screen.blit(font.render(f"Время: {elapsed_time:.2f} сек", True, const.BLACK),
                         (text_x1, text_y1))
        self.screen.blit(font.render(f"Очки: {int(self.total_reward)}", True, const.BLACK),
                         (text_x1, text_y2))
        self.screen.blit(font.render(f"Шаги: {self.step_count}", True, const.BLACK),
                         (text_x1, text_y3))

        self.screen.blit(font.render(f"Обнаружено препятствий: {len(self.known_obstacles)}/{const.COUNT_OBSTACLES}",
                                     True, const.BLACK), (text_x2, text_y1))
        self.screen.blit(
            font.render(f"Обнаружено цветков: {len(self.known_targets)}/{const.COUNT_TARGETS}",
                        True, const.BLACK), (text_x2, text_y2))
        self.screen.blit(font.render(f"Полито цветков: {int(np.sum(self.done_status))}/"
                                     f"{const.COUNT_TARGETS}", True, const.BLACK),
                         (text_x2, text_y3))

        pygame.display.flip()
        pygame.time.wait(50)

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        unavailable_positions = {self.base_position}
        self.target_positions = self._get_objects_positions(unavailable_positions, const.COUNT_TARGETS)
        unavailable_positions.update(self.target_positions)
        self.obstacle_positions = self._get_objects_positions(unavailable_positions, const.COUNT_OBSTACLES)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = const.FIXED_TARGET_POSITIONS.copy()
        self.obstacle_positions = const.FIXED_OBSTACLE_POSITIONS.copy()
        
    def _get_random_base_position(self):
        x = random.randint(0, self.grid_size - self.base_size)
        y = random.randint(0, self.grid_size - self.base_size)
        return (x, y)    
