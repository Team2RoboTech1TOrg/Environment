from collections import deque
import numpy as np
import gymnasium as gym

import logging

from gymnasium import spaces

import const
from AgentObservationSpace import AgentObservationSpace


class Agent:
    def __init__(self, env, name=None):
        self.name = name or f'agent_{id(self)}'
        self.env = env
        self.position = None  # Стартовая позиция
        self.water_tank = None  # Заполняем бак водой
        self.energy = None  # Полный заряд энергии
        self.position_history = None
        self.action_space = spaces.Discrete(const.COUNT_ACTIONS)
        # self.observation_space = spaces.Box( #old
        #     low=-self.env.grid_size,
        #     high=self.env.grid_size,
        #     shape=(2,),
        #     dtype=np.float32
        # )
        self.observation_space = AgentObservationSpace(self.env.grid_size) #new

    def reset(self):
        self.position = self.env.base_position
        self.position_history = deque(maxlen=10)
        self.water_tank = const.WATER_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        return { #new
            'pos': self.position,
            'coords': np.zeros((self.env.grid_size, self.env.grid_size), dtype=np.int32)
        }

        # return self.position #old

    def take_action(self, action):
        reward = 0
        terminated = False
        truncated = False

        # должен возвращать обсервейшн, доделать типы
        if self.energy < 10:  # костыль
            return self.position, reward, True, False, {}

        # пока никак не используется
        obs = self.get_observation()

        if self.water_tank <= 10:  # возврат на базу за водой
            self.position = self.env.base_position
            self.water_tank = const.WATER_CAPACITY

        # Действия агента в зависимости от выбранного действия
        match action:
            case 0:  # Вверх
                new_position = (max(0, self.position[0] - 1), self.position[1])
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 1:  # Вниз
                new_position = (min(const.GRID_SIZE - 1, self.position[0] + 1), self.position[1])
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 2:  # Влево
                new_position = (self.position[0], max(0, self.position[1] - 1))
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 3:  # Вправо
                new_position = (self.position[0], min(const.GRID_SIZE - 1, self.position[1] + 1))
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case _:
                new_position = self.position
        new_position, reward = self.update_visited_cells(new_position)
        self.position = new_position
        logging.info(f"Действие: {action} - позиция: {self.position} - {self.name}")
        return self.position, reward, terminated, truncated, {}

    def get_observation(self):
        for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    pos = (x, y)
                    self.env.viewed_cells.add(pos)
                    if pos in self.env.obstacle_positions:
                        if pos not in self.env.known_obstacles:
                            self.env.known_obstacles.add(pos)
                            logging.debug(f"Новое известное препятствие: {pos}")
                    elif pos in self.env.target_positions:
                        if pos not in self.env.known_flowers:
                            self.env.known_flowers.add(pos)
                            logging.debug(f"Новое известное растение: {pos}")

        # observation = np.array(self.position, dtype=int) #old
        observation =  { #new
            'pos': self.position,
            'coords': np.zeros((self.env.grid_size, self.env.grid_size), dtype=np.int32)
        }

        return observation

    def __repr__(self):
        return f'<Agent {self.name}>'

    def update_visited_cells(self, new_position: tuple[int, int]) -> tuple[tuple[int, int], int]:
        """
        Update explored cells, update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y) and agent reward
        """
        agent_reward = 0
        # Запись истории позиций для обнаружения циклов
        self.position_history.append(new_position)

        # Проверка на выход за границы внутреннего поля
        if not ((self.env.margin <= new_position[0] <= self.env.inner_grid_size) and (
                self.env.margin <= new_position[1] <= self.env.inner_grid_size)):
            agent_reward -= const.PENALTY_OUT_FIELD
            logging.warning(f"Агент вышел за границы внутреннего поля: {new_position}")
            new_position = self.position
        else:
            if new_position in self.env.obstacle_positions:
                agent_reward -= const.PENALTY_HOLE
                new_position = self.position
            elif new_position not in self.env.explored_cells:
                agent_reward += const.REWARD_EXPLORE
                logging.info("Зашел на новую клетку")
                self.env.explored_cells.add(new_position)
                if new_position in self.env.target_positions:
                    self.energy -= const.ENERGY_CONSUMPTION_WATER
                    self.water_tank -= const.WATER_CONSUMPTION
                    idx = self.env.target_positions.index(new_position)
                    self.env.watered_status[idx] = 1
                    logging.info("Опрыскал растение")
            else:
                if len(self.position_history) > 3:
                    if new_position == self.position_history[-2]:
                        agent_reward -= const.PENALTY_LOOP * 3
                        logging.info(f"Штраф за 'стену' {self.position_history[-2]}")
                    elif self.position_history.count(new_position) > 2:
                        agent_reward -= const.PENALTY_LOOP * 2
                        logging.info(f"Штраф за вторичное посещение {new_position} в последние 10 шагов")
                # else: # неудачно работает, подумать
                #     agent_reward -= const.PENALTY_LOOP
                #     logging.info("Штраф за вторичное посещение клетки")
        return new_position, agent_reward
