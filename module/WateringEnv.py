import time
from collections import deque, Counter
import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Agent import Agent
from logger import logging
import const
from utils import convert_to_multidiscrete


class WateringEnv(gym.Env):
    def __init__(self):
        super(WateringEnv, self).__init__()
        self.grid_size = const.GRID_SIZE
        self.margin = const.MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + 120))
        self.base_position = (const.BASE_COORD, const.BASE_COORD)
        self.num_agents = const.NUM_AGENTS
        self.agents = [Agent(self, name=f'agent_{i}') for i in range(self.num_agents)]
        self.start_time = None
        self.reward = None
        self.watered_status = None
        self.step_count = None
        self.position_history = None
        self.action_history = None
        self.known_holes = None
        self.known_flowers = None
        self.viewed_cells = None
        self.explored_cells = None
        action_spaces = spaces.Dict({
            f'agent_{i}': agent.action_space
            for i, agent in enumerate(self.agents)
        })
        self.action_space = convert_to_multidiscrete(action_spaces)
        observation_spaces = {
            f'agent_{i}': agent.observation_space
            for i, agent in enumerate(self.agents)
        }
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed=None, options=None):
        self.reset_objects_positions()
        self.watered_status = np.zeros(const.COUNT_FLOWERS)
        self.start_time = time.time()
        self.reward = 0
        self.step_count = 0
        self.position_history = deque(maxlen=10)
        self.action_history = deque(maxlen=5)
        self.action_history.clear()
        self.known_holes = set()
        self.known_flowers = set()
        self.viewed_cells = set()
        self.explored_cells = set()
        logging.info("Перезагрузка среды")
        obs = {f"agent_{i}": agent.reset() for i, agent in enumerate(self.agents)}
        return obs, {}

    def step(self, actions):
        obs = {}  # new
        self.step_count += 1

        # if self.agent.energy < 10:  # костыль
        #     return self._get_observation(), -100, True, False, {}

        # Добавили действие в общую историю
        self.action_history.append(actions)

        for i, agent in enumerate(self.agents):  # new
            new_position = agent.take_action(actions[i])
            obs[f"agent_{i}"] = new_position
            new_position = self.check_crash(obs, agent, new_position)
            new_position = self.update_visited_cells(new_position, agent)
            agent.position = new_position
            logging.info(
                f"Шаг: {self.step_count},"
                f"Действие: {actions[i]} - позиция: {agent.position} - {agent.name}")
        terminated, truncated, info = self._check_termination_conditions()

        logging.info(
            f"Шаг: {self.step_count},"
            # f"Действие: {actions[i]} - {agent.position}, "
            f"Награда: {self.reward}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )
        return obs, self.reward, terminated, truncated, {}

    def check_crash(self, obs: dict, agent: Agent, new_position: tuple[int, int]):
        """
        Check if agents positions is same.
        :param new_position:
        :param agent:
        :param obs:
        :return: agent coordinates x, y
        """
        crashes = {pos for pos, count in Counter(obs.values()).items() if count > 1}
        if crashes:
            self.reward -= const.PENALTY_CRASH
            logging.warning(f"{crashes} агентов")
            new_position = agent.position
        return new_position

    def update_visited_cells(self, new_position: tuple[int, int], agent: Agent) -> tuple[int, int]:
        """
        Update explored cells, update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param agent:
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y)
        """
        # Запись истории позиций для обнаружения циклов
        self.position_history.append(new_position)

        # Проверка на выход за границы внутреннего поля
        if not ((self.margin <= new_position[0] <= self.inner_grid_size) and (
                self.margin <= new_position[1] <= self.inner_grid_size)):
            self.reward -= const.PENALTY_OUT_FIELD
            logging.warning(f"Агент вышел за границы внутреннего поля: {new_position}")
            new_position = agent.position
        else:
            if new_position in self.hole_positions:
                self.reward -= const.PENALTY_HOLE
                new_position = agent.position
            elif new_position not in self.explored_cells:
                self.reward += const.REWARD_EXPLORE
                logging.info("Зашел на новую клетку")
                self.explored_cells.add(new_position)
                if new_position in self.target_positions:
                    agent.energy -= const.ENERGY_CONSUMPTION_WATER
                    agent.water_tank -= const.WATER_CONSUMPTION
                    idx = self.target_positions.index(new_position)
                    self.watered_status[idx] = 1
                    logging.info("Полил цветок")
            else:
                if new_position == self.position_history[-2]:
                    self.reward -= const.PENALTY_LOOP * 3
                    logging.info(f"Штраф за 'стену' {self.position_history[-2]}")
                elif self.position_history.count(new_position) > 2:
                    self.reward -= const.PENALTY_LOOP * 2
                    logging.info("Штраф за вторичное посещение клетки в последние 10 шагов")
                # else: # неудачно работает, подумать
                #     self.reward -= const.PENALTY_LOOP
                #     logging.info("Штраф за вторичное посещение клетки")
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
            truncated = True

        elif np.all(self.watered_status == 1):
            logging.info("Все цветы политы")
            for agent in self.agents:
                agent.position = self.base_position
            logging.info("Агенты вернулись на базу")
            # условие по времени выполнения
            if self.step_count <= const.MIN_GAME_STEPS:
                self.reward += const.REWARD_COMPLETION * 3
            else:
                self.reward += const.REWARD_COMPLETION
            terminated = True

        return terminated, truncated, {}

    def render(self):
        """Render agent game"""
        self.screen.fill(const.GREEN)
        # Отрисовка сетки
        for x in range(self.grid_size):  # self.margin, self.grid_size - self.margin):
            for y in range(self.grid_size):  # self.margin, self.grid_size - self.margin):
                pygame.draw.rect(
                    self.screen, const.BLACK,
                    (x * const.CELL_SIZE, y * const.CELL_SIZE, const.CELL_SIZE, const.CELL_SIZE), 1
                )

        # Отрисовка границы внутреннего поля (устанавливаем цвет и толщину линии)
        inner_field_size = self.inner_grid_size * const.CELL_SIZE
        margin_x = (self.grid_size * const.CELL_SIZE - inner_field_size) // 2
        margin_y = (self.grid_size * const.CELL_SIZE - inner_field_size) // 2
        inner_field_rect = pygame.Rect(margin_x, margin_y, inner_field_size, inner_field_size)
        pygame.draw.rect(self.screen, const.BLACK, inner_field_rect, 4)

        # Отрисовка базы
        self.screen.blit(const.BASE_ICON,
                         (self.base_position[1] * const.CELL_SIZE, self.base_position[0] * const.CELL_SIZE))

        # Рисуем цветы и ямы, которые были обнаружены
        for i, pos in enumerate(self.target_positions):
            if pos in self.known_flowers:
                if self.watered_status[i]:
                    icon = const.WATERED_FLOWER_ICON
                else:
                    icon = const.FLOWER_ICON
                self.screen.blit(icon, (pos[1] * const.CELL_SIZE, pos[0] * const.CELL_SIZE))

        for hole in self.hole_positions:
            if hole in self.known_holes:
                self.screen.blit(const.HOLE_ICON, (hole[1] * const.CELL_SIZE, hole[0] * const.CELL_SIZE))

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if pos not in self.viewed_cells:
                    dark_overlay = pygame.Surface((const.CELL_SIZE, const.CELL_SIZE), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))  # Непрозрачный
                    self.screen.blit(dark_overlay, (y * const.CELL_SIZE, x * const.CELL_SIZE))

        # Отрисовка времени, очков, заряда и уровня воды
        elapsed_time = time.time() - self.start_time  # Рассчитываем время
        font = pygame.font.SysFont(None, const.FONT_SIZE)
        status_bar_height = 120  # Высота панели статуса
        status_bar_rect = pygame.Rect(0, const.SCREEN_SIZE, const.SCREEN_SIZE,
                                      status_bar_height)  # Прямоугольник для панели статуса
        pygame.draw.rect(self.screen, const.WHITE, status_bar_rect)

        self.screen.blit(font.render(f"Время: {elapsed_time:.2f} сек", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Очки: {int(self.reward)}", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 40))
        # self.screen.blit(font.render(f"Энергия: {self.agent.energy}", True, const.BLACK),
        #                  (200, const.SCREEN_SIZE + 10))
        # self.screen.blit(font.render(f"Вода: {self.agent.water_tank}", True, const.BLACK),
        #                  (200, const.SCREEN_SIZE + 40))
        self.screen.blit(font.render(f"Шаги: {self.step_count}", True, const.BLACK),
                         (400, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Обнаружено ям: {len(self.known_holes)}/{const.COUNT_HOLES}",
                                     True, const.BLACK), (400, const.SCREEN_SIZE + 40))
        self.screen.blit(
            font.render(f"Обнаружено цветков: {len(self.known_flowers)}/{const.COUNT_FLOWERS}",
                        True, const.BLACK), (10, const.SCREEN_SIZE + 70))
        self.screen.blit(font.render(f"Полито цветков: {int(np.sum(self.watered_status))}/"
                                     f"{const.COUNT_FLOWERS}", True, const.BLACK), (300, const.SCREEN_SIZE + 70))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(const.AGENT_ICON, (agent.position[1] * const.CELL_SIZE,
                                                agent.position[0] * const.CELL_SIZE))

        pygame.display.flip()
        pygame.time.wait(10)

    def render_message(self, render_text: str):
        """
        Display message in the center of screen
        :param render_text: str
        :return:
        """
        self.screen.fill(const.BLACK)
        text_surf = pygame.font.SysFont(None, const.TITLE_SIZE).render(render_text, True, const.GREEN)
        self.screen.blit(text_surf, text_surf.get_rect(center=(const.SCREEN_SIZE // 2, const.SCREEN_SIZE // 2)))
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
        unavailable_positions = {self.base_position}
        self.target_positions = self._get_objects_positions(unavailable_positions, const.COUNT_FLOWERS)
        unavailable_positions.update(self.target_positions)
        self.hole_positions = self._get_objects_positions(unavailable_positions, const.COUNT_HOLES)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = const.FIXED_FLOWER_POSITIONS.copy()
        self.hole_positions = const.FIXED_HOLE_POSITIONS.copy()

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
