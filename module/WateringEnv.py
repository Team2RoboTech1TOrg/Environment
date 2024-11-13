import time
from collections import deque, Counter
import pygame
import gymnasium as gym
import numpy as np

from Agent import Agent
from logger import logging
import const
from utils import convert_to_multidiscrete, load_image


class WateringEnv(gym.Env):
    def __init__(self, num_agents: int, grid_size: int):
        super(WateringEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = const.SCREEN_SIZE // self.grid_size
        self.margin = const.MARGIN_SIZE
        self.inner_grid_size = self.grid_size - self.margin * 2
        self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + 120))
        self.base_position = (self.grid_size // 2, self.grid_size // 2)
        self.num_agents = num_agents
        self.agents = [Agent(self, name=f'agent_{i}') for i in range(self.num_agents)]
        self.start_time = None
        self.total_reward = None
        self.step_reward = None
        self.watered_status = None
        self.step_count = None
        self.position_history = None
        self.action_history = None
        self.known_obstacles = None
        self.known_flowers = None
        self.viewed_cells = None
        self.explored_cells = None
        action_spaces = gym.spaces.Dict({
            f'agent_{i}': agent.action_space
            for i, agent in enumerate(self.agents)
        })
        self.action_space = convert_to_multidiscrete(action_spaces)

        self.observation_space = gym.spaces.Dict({  # new, old - coords
            'coords': gym.spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            'pos': gym.spaces.Box(
                low=np.stack([agent.observation_space.position_space.low for agent in self.agents], axis=0),
                high=np.stack([agent.observation_space.position_space.high for agent in self.agents], axis=0),
                shape=(self.num_agents, 2),
                dtype=np.int32),
        })

        print(self.observation_space)

    def reset(self, *, seed=None, options=None):
        self.reset_objects_positions()
        self.watered_status = np.zeros(const.COUNT_FLOWERS)
        self.start_time = time.time()
        self.total_reward = 0
        self.step_reward = 0
        self.step_count = 0
        self.action_history = deque(maxlen=5)
        self.action_history.clear()
        self.known_obstacles = set()
        self.known_flowers = set()
        self.viewed_cells = set()
        self.explored_cells = set()
        logging.info("Перезагрузка среды")
        # obs = {'pos': np.stack([agent.reset() for agent in self.agents], axis=0)}
        agent_obs = [agent.reset() for agent in self.agents]
        positions = np.stack([obs['pos'] for obs in agent_obs])
        coords = np.max(np.stack([obs['coords'] for obs in agent_obs]), axis=0)
        obs = {'pos': positions, 'coords': coords}  # new
        return obs, {}  # old

    def step(self, actions):
        logging.info(f"Шаг: {self.step_count}")

        agent_obs = [agent.get_observation() for agent in self.agents]
        obs = {'pos': np.stack([obs['pos'] for obs in agent_obs]),
               'coords': np.max(np.stack([obs['coords'] for obs in agent_obs]), axis=0)}

        self.step_reward = 0
        self.step_count += 1

        # Добавили действие в общую историю
        self.action_history.append(actions)

        for i, agent in enumerate(self.agents):
            new_position, agent_reward, terminated, truncated, info = agent.take_action(actions[i])
            # new_position = self.check_crash(obs, agent, new_position)
            obs['pos'][i] = new_position  # new ??
            self.step_reward += agent_reward
        reward, terminated, truncated, info = self._check_termination_conditions()

        logging.info(
            f"Награда: {self.total_reward}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )
        return obs, reward, terminated, truncated, {}

    def check_crash(self, obs: dict, agent: Agent, new_position: tuple[int, int]):
        """
        Check if agents positions is same.
        :param new_position: position of agent (x, y)
        :param agent: agent in process
        :param obs: all agents positions at the moment
        :return: agent coordinates x, y
        """
        crashes = {pos for pos, count in Counter(obs.values()).items() if count > 1}
        if crashes:
            self.total_reward -= const.PENALTY_CRASH
            logging.warning(f"Столкнование {crashes} агентов")
            new_position = agent.position
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

        elif np.all(self.watered_status == 1):
            terminated = True
            logging.info("Все растения опрысканы")
            for agent in self.agents:
                agent.position = self.base_position
            logging.info("Агенты вернулись на базу")

            # условие по времени выполнения
            if self.step_count <= const.MIN_GAME_STEPS:
                total_reward = self.total_reward + const.REWARD_COMPLETION * 3
            else:
                total_reward = self.total_reward + const.REWARD_COMPLETION
            logging.info(f"Награда: {total_reward}")
            self.total_reward = 0
        else:
            self.total_reward += self.step_reward
            total_reward = 0
        return total_reward, terminated, truncated, {}

    def render(self):
        """Render agent game"""
        AGENT_ICON = load_image(const.AGENT, self.cell_size)  # Изображение робота
        FLOWER_ICON = load_image(const.FLOWER, self.cell_size)  # Сухие цветы
        WATERED_FLOWER_ICON = load_image(const.WATERED, self.cell_size)  # Политые цветы
        OBSTACLE_ICON = load_image(const.OBSTACLE, self.cell_size)  # Яма
        BASE_ICON = load_image(const.BASE, self.cell_size)  # База

        self.screen.fill(const.GREEN)
        # Отрисовка сетки
        for x in range(self.grid_size):  # self.margin, self.grid_size - self.margin):
            for y in range(self.grid_size):  # self.margin, self.grid_size - self.margin):
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

        # Отрисовка базы
        self.screen.blit(BASE_ICON,
                         (self.base_position[1] * self.cell_size, self.base_position[0] * self.cell_size))

        # Рисуем цветы и ямы, которые были обнаружены
        for i, pos in enumerate(self.target_positions):
            if pos in self.known_flowers:
                if self.watered_status[i]:
                    icon = WATERED_FLOWER_ICON
                else:
                    icon = FLOWER_ICON
                self.screen.blit(icon, (pos[1] * self.cell_size, pos[0] * self.cell_size))

        for hole in self.obstacle_positions:
            if hole in self.known_obstacles:
                self.screen.blit(OBSTACLE_ICON, (hole[1] * self.cell_size, hole[0] * self.cell_size))

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if pos not in self.viewed_cells:
                    dark_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))  # Непрозрачный
                    self.screen.blit(dark_overlay, (y * self.cell_size, x * self.cell_size))

        # Отрисовка времени, очков, заряда и уровня воды
        elapsed_time = time.time() - self.start_time  # Рассчитываем время
        font = pygame.font.SysFont(None, const.FONT_SIZE)
        status_bar_height = 120  # Высота панели статуса
        status_bar_rect = pygame.Rect(0, const.SCREEN_SIZE, const.SCREEN_SIZE,
                                      status_bar_height)  # Прямоугольник для панели статуса
        pygame.draw.rect(self.screen, const.WHITE, status_bar_rect)

        self.screen.blit(font.render(f"Время: {elapsed_time:.2f} сек", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Очки: {int(self.total_reward)}", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 40))
        # self.screen.blit(font.render(f"Энергия: {self.agent.energy}", True, const.BLACK),
        #                  (200, const.SCREEN_SIZE + 10))
        # self.screen.blit(font.render(f"Вода: {self.agent.water_tank}", True, const.BLACK),
        #                  (200, const.SCREEN_SIZE + 40))
        self.screen.blit(font.render(f"Шаги: {self.step_count}", True, const.BLACK),
                         (400, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Обнаружено ям: {len(self.known_obstacles)}/{const.COUNT_OBSTACLES}",
                                     True, const.BLACK), (400, const.SCREEN_SIZE + 40))
        self.screen.blit(
            font.render(f"Обнаружено цветков: {len(self.known_flowers)}/{const.COUNT_FLOWERS}",
                        True, const.BLACK), (10, const.SCREEN_SIZE + 70))
        self.screen.blit(font.render(f"Полито цветков: {int(np.sum(self.watered_status))}/"
                                     f"{const.COUNT_FLOWERS}", True, const.BLACK), (300, const.SCREEN_SIZE + 70))

        # Отрисовка агента
        for agent in self.agents:
            self.screen.blit(AGENT_ICON, (agent.position[1] * self.cell_size,
                                          agent.position[0] * self.cell_size))

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
        self.obstacle_positions = self._get_objects_positions(unavailable_positions, const.COUNT_OBSTACLES)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = const.FIXED_FLOWER_POSITIONS.copy()
        self.obstacle_positions = const.FIXED_HOLE_POSITIONS.copy()

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
