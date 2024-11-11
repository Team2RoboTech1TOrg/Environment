import numpy as np
import gymnasium as gym

import logging
import const


class Agent:
    def __init__(self, env):
        self.env = env
        self.position = None  # Стартовая позиция
        self.water_tank = None  # Заполняем бак водой
        self.energy = None  # Полный заряд энергии
        self.action_space = gym.spaces.Discrete(const.COUNT_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-self.env.grid_size,
            high=self.env.grid_size,
            shape=(2,),
            dtype=np.float32
        )

    def reset(self):
        self.position = self.env.base_position
        self.water_tank = const.WATER_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        return self.position

    def take_action(self, action):
        # Обновляем наблюдение, чтобы получить новые расстояния
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
        return new_position #new old: obs, pos

    def get_observation(self):
        for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    pos = (x, y)
                    self.env.viewed_cells.add(pos)
                    if pos in self.env.hole_positions:
                        if pos not in self.env.known_holes:
                            self.env.known_holes.add(pos)
                            logging.debug(f"Новая известная яма: {pos}")
                    elif pos in self.env.target_positions:
                        if pos not in self.env.known_flowers:
                            self.env.known_flowers.add(pos)
                            logging.debug(f"Новый известный цветок: {pos}")

        observation = np.array(self.position, dtype=int)
        return observation
