import random
import numpy as np

from collections import deque
from gymnasium import spaces
from typing import Any

import logging

from numpy import ndarray

import const
from spaces.AgentObservationSpace import AgentObservationSpace
from PointStatus import PointStatus, ObjectStatus


class Agent:
    def __init__(self, scenario, name=None):
        self.dinamic_coef = None
        self.reward_coef = None
        self.name = name or id(self)
        self.env = scenario
        self.position = None
        self.explorator = False
        self.tank = None
        self.energy = None
        self.position_history = None
        self.action_space = spaces.Discrete(const.COUNT_ACTIONS)
        self.observation_space = AgentObservationSpace(self.env.grid_size)

    def reset(self):
        if self.env.name == 'exploration':
            self.explorator = True
        self.position = random.choice(self.env.base_positions)
        self.reward_coef = 1
        self.dinamic_coef = 1.03
        self.position_history = deque(maxlen=10)#self.env.inner_grid_size)
        self.tank = const.TANK_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        coords = np.zeros((self.env.grid_size, self.env.grid_size, 2), dtype=np.int32)
        logging.info(f"Позиция {self.name} стартовая {self.position}")
        return {
            'pos': self.position,
            'coords': coords
        }

    def take_action(self, action):
        reward = 0
        terminated = False
        truncated = False

        if self.energy < 10:
            self.position = random.choice(self.env.base_positions)
            return self.position, reward, False, True, {}

        if not self.explorator:
            if self.tank < 10:  # возврат на базу
                self.position = random.choice(self.env.base_positions)
                self.tank = const.TANK_CAPACITY

        obs = self.get_observation()

        # Действия агента в зависимости от выбранного действия
        match action:
            case 0:  # Вверх
                new_position = (max(0, self.position[0] - 1), self.position[1])
            case 1:  # Вниз
                new_position = (min(self.env.grid_size - 1, self.position[0] + 1), self.position[1])
            case 2:  # Влево
                new_position = (self.position[0], max(0, self.position[1] - 1))
            case 3:  # Вправо
                new_position = (self.position[0], min(self.env.grid_size - 1, self.position[1] + 1))
            case 4:  # На месте
                new_position = self.position
            case _:
                new_position = self.position
        self.energy -= const.ENERGY_CONSUMPTION_MOVE

        value_new_position = obs['coords'][new_position[0]][new_position[1]]
        # если не таргет, отмечаем посещение клетки
        if value_new_position[1] != ObjectStatus.plant.value:
            obs['coords'][new_position[0]][new_position[1]][0] = PointStatus.visited.value
        new_position, reward = self.get_agent_rewards(new_position, value_new_position[1], action)
        self.position = new_position

        logging.info(f"{self.name} действие: {action} - позиция: {self.position}")
        return self.position, reward, terminated, truncated, {}

    def get_observation(self) -> dict[str, ndarray]:
        coords = np.full((self.env.grid_size, self.env.grid_size, 2), fill_value=0)
        for pos in self.get_review():
            x, y = pos
            coords[x][y][0] = PointStatus.viewed.value
            if pos in self.env.obstacle_positions:
                coords[x][y][1] = ObjectStatus.obstacle.value
            elif pos in self.env.plants_positions:
                coords[x][y][1] = ObjectStatus.plant.value

        observation = {
            'pos': self.position,
            'coords': coords
        }
        return observation

    def __repr__(self):
        return f'{self.name}'

    def get_agent_rewards(self, new_position: tuple[int, int], value: float, action: Any) -> tuple[
            tuple[int, int], int]:
        """
        Update explored cells, update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param action: chosen action
        :param value: status and objects in agent position
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y) and agent reward
        """
        agent_reward = 0
        # не хранить если действие - стоять на месте
        if action != 4:
            self.position_history.append(new_position)

        if len(self.position_history) > 3:
            agent_reward += self.check_loop(new_position)

        if not ((self.env.margin <= new_position[0] <= self.env.inner_grid_size) and (
                self.env.margin <= new_position[1] <= self.env.inner_grid_size)):
            agent_reward -= const.PENALTY_OUT_FIELD
            logging.warning(f"{self} вышел за границы внутреннего поля: {new_position}")
            new_position = self.position
        elif value == ObjectStatus.obstacle.value:
            agent_reward -= const.PENALTY_OBSTACLE
            new_position = self.position
            logging.info(
                f"Упс, препятствие! {self} - штраф {const.PENALTY_OBSTACLE}, вернулся на {new_position}")
        elif value == ObjectStatus.plant.value:
            if not self.explorator:
                idx = self.env.target_positions.index(new_position)
                if self.env.done_status[idx] == 0:
                    self.energy -= const.ENERGY_CONSUMPTION_DONE
                    self.tank -= const.ON_TARGET_CONSUMPTION
                    self.env.done_status[idx] = 1
                    agent_reward += const.REWARD_DONE
                    logging.info(f"{self} выполнена задача {new_position}, награда {round(agent_reward, 2)}")

        return new_position, agent_reward

    def check_loop(self, new_position) -> int:
        """
        Calculate penalty for loop for agent position
        :param new_position: coordinates of agent (x, y)
        :return: penalty for loop
        """
        reward = 0
        pos_counter = self.position_history.count(new_position)
        if new_position == self.position_history[-2]:
            reward -= const.PENALTY_LOOP
            logging.warning(f"Штраф {self} за второй раз в одну клетку' {self.position_history[-2]}")
        elif 4 >= pos_counter > 2:
            reward -= const.PENALTY_LOOP * 2
            logging.warning(
                f"Штраф {self} за вторичное посещение {new_position}"
                f" в последние {len(self.position_history)} шагов")
        elif pos_counter > 4:
            reward -= const.PENALTY_LOOP * 5
            logging.warning(
                f"Штраф {self} за мнократное посещение {new_position}"
                f" в последние {len(self.position_history)} шагов")
        return reward

    def get_review(self) -> list[tuple[int, int]]:
        review = []
        view = const.VIEW_RANGE
        for dx in range(-view, view + 1):
            for dy in range(-view, view + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    review.append((x, y))
        return review
