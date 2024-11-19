import random
from typing import Any

import numpy as np

from collections import deque
from gymnasium import spaces

import logging
import const
from spaces.AgentObservationSpace import AgentObservationSpace
from PointStatus import PointStatus, ObjectStatus


class Agent:
    def __init__(self, scenario, name=None):
        self.reward_coef = None
        self.name = name or f'agent_{id(self)}'
        self.env = scenario
        self.position = None
        self.tank = None
        self.energy = None
        self.position_history = None
        self.action_space = spaces.Discrete(const.COUNT_ACTIONS)
        self.observation_space = AgentObservationSpace(self.env.grid_size)

    def reset(self):
        self.position = random.choice(self.env.base_positions)
        self.reward_coef = 1
        logging.info(f"Позиция {self.name} стартовая {self.position}")
        self.position_history = deque(maxlen=10)
        self.tank = const.TANK_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        coords = np.zeros((self.env.grid_size, self.env.grid_size, 2), dtype=np.int32)
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

        if self.tank <= 10:  # возврат на базу
            self.position = random.choice(self.env.base_positions)
            self.tank = const.TANK_CAPACITY

        obs = self.get_observation()

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
            case 4:  # На месте
                new_position = self.position
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case _:
                new_position = self.position

        value_new_position = obs['coords'][new_position[0]][new_position[1]]
        new_position, reward = self.get_agent_rewards(new_position, value_new_position, action)
        self.position = new_position
        logging.info(f"{self.name} действие: {action} - позиция: {self.position}")
        return self.position, reward, terminated, truncated, {}

    def get_observation(self):
        coords = np.full((self.env.grid_size, self.env.grid_size, 2), fill_value=0)
        for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    pos = (x, y)
                    coords[x][y][0] = PointStatus.viewed.value
                    if pos in self.env.obstacle_positions:
                        coords[x][y][1] = ObjectStatus.obstacle.value
                    elif pos in self.env.target_positions:
                        coords[x][y][1] = ObjectStatus.target.value

        observation = {
            'pos': self.position,
            'coords': coords
        }
        return observation

    def __repr__(self):
        return f'<Agent {self.name}>'

    def get_agent_rewards(self, new_position: tuple[int, int], value: float, action: Any) -> tuple[
            tuple[int, int], int]:
        """
        Update explored cells, update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param action:
        :param value:
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y) and agent reward
        """
        agent_reward = 0
        # не хранить если действие - стоять на месте
        if action != 4:
            self.position_history.append(new_position)

        if not ((self.env.margin <= new_position[0] <= self.env.inner_grid_size) and (
                self.env.margin <= new_position[1] <= self.env.inner_grid_size)):
            agent_reward -= const.PENALTY_OUT_FIELD
            logging.warning(f"Агент {self.name} вышел за границы внутреннего поля: {new_position}")
            new_position = self.position
        else:
            if value[1] == ObjectStatus.obstacle.value:
                agent_reward -= const.PENALTY_OBSTACLE
                new_position = self.position
                logging.info(
                    f"Упс, препятствие! {self.name} - штраф {const.PENALTY_OBSTACLE}, вернулся на {new_position}")
            elif value[1] == ObjectStatus.target.value:
                idx = self.env.target_positions.index(new_position)
                if self.env.done_status[idx] == 0:
                    self.energy -= const.ENERGY_CONSUMPTION_DONE
                    self.tank -= const.ON_TARGET_CONSUMPTION
                    self.env.done_status[idx] = 1
                    self.reward_coef *= 1.03
                    reward = const.REWARD_DONE * self.reward_coef
                    agent_reward += reward
                    logging.info(f"{self.name} выполнена задача {new_position}, награда {round(reward, 2)}")
            else:
                if len(self.position_history) > 3:
                    pos_counter = self.position_history.count(new_position)
                    if new_position == self.position_history[-2]:
                        agent_reward -= const.PENALTY_LOOP * 2
                        logging.info(f"Штраф {self.name} за второй раз в одну клетку' {self.position_history[-2]}")
                    elif 4 > pos_counter > 2:
                        agent_reward -= const.PENALTY_LOOP * 3
                        logging.info(f"Штраф {self.name} за вторичное посещение {new_position} в последние 10 шагов")
                    elif pos_counter >= 4:
                        agent_reward -= const.PENALTY_LOOP * 5
                        logging.info(f"Штраф {self.name} за мнократное посещение {new_position} в последние 10 шагов")
        return new_position, agent_reward
