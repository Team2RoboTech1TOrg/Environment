import numpy as np

from collections import deque, Counter
from gymnasium import spaces
import random # для костыля keno
import logging
import const
from AgentObservationSpace import AgentObservationSpace
from PointStatus import PointStatus, ObjectStatus


class Agent:
    def __init__(self, scenario, name=None):
        self.name = name or f'agent_{id(self)}'
        self.env = scenario
        # TO DO позиции рандомно из базы
        self.position = None
        self.tank = None
        self.energy = None
        self.action_space = spaces.Discrete(const.COUNT_ACTIONS)
        self.observation_space = AgentObservationSpace(self.env.grid_size)
        self.current_steps = 0
        self.position_history = deque(maxlen=10)
        self.position_counts = Counter()
        self.previous_distance = float('inf')

    def reset(self):
        base_x, base_y = self.env.base_position
        self.position = (base_x, base_y + self.env.base_size - 1)
        self.position = self.env.base_position
        self.tank = const.TANK_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        self.position_history.clear()
        self.position_counts.clear()
        coords = np.zeros((self.env.grid_size, self.env.grid_size, 2), dtype=np.int32)
        self.current_steps = 0
        self.previous_distance = float('inf')
        return {
            'pos': self.position,
            'coords': coords
        }

    def find_nearest_unexplored(self):
        min_distance = float('inf')
        target_position = None
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                if self.env.current_map[x, y, 0] == PointStatus.empty.value:
                    distance = abs(self.position[0] - x) + abs(self.position[1] - y)
                    if distance < min_distance:
                        min_distance = distance
                        target_position = (x, y)
        return target_position, min_distance

    # def find_nearest_unwatered_flower(self):
        # min_distance = float('inf')
        # target_position = None
        # for idx, pos in enumerate(self.env.target_positions):
            # if self.env.done_status[idx] == 0:
                # distance = abs(self.position[0] - pos[0]) + abs(self.position[1] - pos[1])
                # if distance < min_distance:
                    # min_distance = distance
                    # target_position = pos
        # return target_position, min_distance
        
    def find_nearest_unwatered_flower(self):
        min_distance = float('inf')
        target_position = None
        # Используем общую карту известных неполитых цветков
        known_targets = np.argwhere((self.env.current_map[:, :, 1] == ObjectStatus.target.value) &
                                    (self.env.current_map[:, :, 2] == 0))  # Флаг "полит ли цветок"

        for pos in known_targets:
            pos = tuple(pos)
            distance = abs(self.position[0] - pos[0]) + abs(self.position[1] - pos[1])
            if distance < min_distance:
                min_distance = distance
                target_position = pos
        return target_position, min_distance
        
    def take_action(self, action):
        reward = -const.STEP_PENALTY  # Штраф за шаг
        terminated = False
        truncated = False

        if self.energy <= 0:
            terminated = True
            reward -= const.PENALTY_NO_ENERGY
            return self.position, reward, terminated, truncated, {}

        obs = self.get_observation()

        if self.tank <= 0:
            # Возврат на базу для заправки
            self.position = self.env.base_position
            self.tank = const.TANK_CAPACITY

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
                
        # Проверка на препятствия
        if new_position in self.env.obstacle_positions:
            reward -= const.PENALTY_OBSTACLE
            new_position = self.position  # Остаемся на месте
            logging.info("Упс, препятствие!")
            
        value_new_position = obs['coords'][new_position[0]][new_position[1]]
 
        new_position, step_reward = self.get_agent_rewards(new_position, value_new_position)
        self.position = new_position
        reward += step_reward
        logging.info(f"Действие: {action} - позиция: {self.position} - {self.name}")
        return self.position, reward, terminated, truncated, {}

    # def get_observation(self):
        # coords = np.full((self.env.grid_size, self.env.grid_size, 2), fill_value=0) #new
        # # coords = np.zeros((self.env.grid_size, self.env.grid_size))
        # for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            # for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                # x, y = self.position[0] + dx, self.position[1] + dy
                # if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    # pos = (x, y)
                    # coords[x][y][0] = PointStatus.viewed.value
                    # # logging.info(f"{self.name} увидел новую клетку {pos}")
                    # if pos in self.env.obstacle_positions:
                        # coords[x][y][1] = ObjectStatus.obstacle.value
                    # elif pos in self.env.target_positions:
                        # coords[x][y][1] = ObjectStatus.target.value

        # observation = {
            # 'pos': self.position,
            # 'coords': coords
        # }
        # return observation
        
    def get_observation(self):
        # Создаём карту наблюдений агента
        agent_coords = np.full((self.env.grid_size, self.env.grid_size, 3), fill_value=0)

        # Агент обновляет карту на основе своей области зрения
        for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    pos = (x, y)
                    agent_coords[x][y][0] = PointStatus.viewed.value
                    if pos in self.env.obstacle_positions:
                        agent_coords[x][y][1] = ObjectStatus.obstacle.value
                    elif pos in self.env.target_positions:
                        idx = self.env.target_positions.index(pos)
                        agent_coords[x][y][1] = ObjectStatus.target.value
                        agent_coords[x][y][2] = self.env.done_status[idx]  # 0 или 1 в зависимости от состояния

        # Возвращаем наблюдения агента без изменения общей карты
        observation = {
            'pos': self.position,
            'coords': agent_coords
        }
        return observation 

    def __repr__(self):
        return f'<Agent {self.name}>'

    def get_agent_rewards(self, new_position: tuple[int, int], value: float) -> tuple[tuple[int, int], int]:
        """
        Update explored cells, update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param value:
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y) and agent reward
        """
        agent_reward = 0

        # Обновление истории позиций и счетчика посещений
        if len(self.position_history) == self.position_history.maxlen:
            oldest_position = self.position_history.popleft()
            self.position_counts[oldest_position] -= 1
            if self.position_counts[oldest_position] == 0:
                del self.position_counts[oldest_position]

        self.position_history.append(new_position)
        self.position_counts[new_position] += 1

        # Проверка на повторные посещения
        if self.position_counts[new_position] > 2:
            agent_reward -= const.PENALTY_LOOP
            logging.info(f"Штраф за повторное посещение {new_position}")
        
        # Проверка на выход за границы внутреннего поля
        if not ((self.env.margin <= new_position[0] <= self.env.inner_grid_size) and (
                self.env.margin <= new_position[1] <= self.env.inner_grid_size)):
            agent_reward -= const.PENALTY_OUT_FIELD
            logging.warning(f"Агент вышел за границы внутреннего поля: {new_position}")
            new_position = self.position
            
        # Проверка препятствий
        if value[1] == ObjectStatus.obstacle.value:
            agent_reward -= const.PENALTY_OBSTACLE
            new_position = self.position
            logging.info("Упс, препятствие!")

        # Проверка на неполитый цветок на текущей позиции
        elif new_position in self.env.target_positions:
            idx = self.env.target_positions.index(new_position)
            if self.env.done_status[idx] == 0:
                self.energy -= const.ENERGY_CONSUMPTION_DONE
                self.tank -= const.ON_TARGET_CONSUMPTION
                self.env.done_status[idx] = 1
                self.env.current_map[new_position[0], new_position[1], 2] = 1  # Обозначаем цветок как политый
                agent_reward += const.REWARD_SPRAY
                logging.info(f"{self.name} опрыскал растение на позиции {new_position}")

        else:
            # Проверяем, есть ли неполитый цветок рядом
            adjacent_positions = [
                (new_position[0] + dx, new_position[1] + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= new_position[0] + dx < self.env.grid_size and 0 <= new_position[1] + dy < self.env.grid_size
            ]
            for pos in adjacent_positions:
                if pos in self.env.target_positions:
                    idx = self.env.target_positions.index(pos)
                    if self.env.done_status[idx] == 0:
                        # Агент прошел мимо неполитого цветка
                        agent_reward -= const.PENALTY_IGNORE_FLOWER
                        logging.info(f"{self.name} прошел мимо неполитого цветка на позиции {pos}")
                        break

        # Поощрение за приближение к неполитым цветкам
        unwatered_flower, distance_to_flower = self.find_nearest_unwatered_flower()
        if unwatered_flower:
            if distance_to_flower < self.previous_distance:
                agent_reward += const.REWARD_APPROACH_TARGET
            elif distance_to_flower > self.previous_distance:
                agent_reward -= const.PENALTY_MOVE_AWAY
            self.previous_distance = distance_to_flower
        else:
            # Если нет неполитых цветков, поощряем исследование
            unexplored_target, distance_to_unexplored = self.find_nearest_unexplored()
            if unexplored_target:
                if distance_to_unexplored < self.previous_distance:
                    agent_reward += const.REWARD_APPROACH_TARGET
                elif distance_to_unexplored > self.previous_distance:
                    agent_reward -= const.PENALTY_MOVE_AWAY
                self.previous_distance = distance_to_unexplored

        return new_position, agent_reward
