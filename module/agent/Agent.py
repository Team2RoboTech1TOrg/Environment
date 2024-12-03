import random
import numpy as np

from collections import deque

import logging
import const as c
from spaces.AgentObservationSpace import AgentObservationSpace
from enums.PointStatus import PointStatus as Point
from enums.ObjectStatus import ObjectStatus as Obj
from enums.DoneStatus import DoneStatus as Done
from enums.ActionsNames import ActionsNames as Act


class Agent:
    def __init__(self, scenario, name=None):
        self.reward_coef = None
        self.name = name or id(self)
        self.env = scenario
        self.position = None
        self.explorator = False
        self.tank = None
        self.energy = None
        self.position_history = None
        self.observation_space = AgentObservationSpace(self.env.grid_size)
        self.view_range = 1
        self.reward_coef = None

    def reset(self):
        if self.env.name == 'exploration':
            self.explorator = True
        else:
            self.tank = c.TANK_CAPACITY
        self.position = random.choice(self.env.base_positions)
        self.reward_coef = 1
        self.position_history = deque(maxlen=10)
        self.energy = c.ENERGY_CAPACITY
        coords = np.zeros((self.env.grid_size, self.env.grid_size, 3), dtype=np.int32)
        logging.info(f"Позиция {self.name} стартовая {self.position}")
        return {
            'pos': self.position,
            'coords': coords
        }

    def take_action(self, action) -> tuple[tuple[int, int], float, bool, bool, dict]:
        """
        Agent get action from model prediction, done step, check conditions and reward of new position.
        :param action: action predict from model
        :return: position of agent, his reward, terminated, truncated, information
        """
        reward = 0
        terminated = False
        truncated = False

        if self.energy < 10:
            self.position = random.choice(self.env.base_positions)
            return self.position, reward, False, True, {}

        if not self.explorator:
            if self.tank < 10:  # возврат на базу
                self.position = random.choice(self.env.base_positions)
                self.tank = c.TANK_CAPACITY
                logging.info(f"{self.name} полетел на базу за пестицидами")

        obs = self.get_observation()

        x, y = self.position
        match action:
            case Act.up.value:
                new_position = (x - 1, y)
            case Act.down.value:
                new_position = (x + 1, y)
            case Act.left.value:
                new_position = (x, y - 1)
            case Act.right.value:
                new_position = (x, y + 1)
            case Act.stop.value:
                new_position = self.position
            case Act.right_up.value:
                new_position = (x - 1, y + 1)
            case Act.left_up.value:
                new_position = (x - 1, y - 1)
            case Act.right_down.value:
                new_position = (x + 1, y + 1)
            case Act.left_down.value:
                new_position = (x + 1, y - 1)
            case _:
                new_position = self.position

        new_position = np.clip(new_position, self.observation_space.position_space.low,
                               self.observation_space.position_space.high)
        self.energy -= c.ENERGY_CONSUMPTION_MOVE

        new_position = tuple(new_position)
        if action != 4:
            self.position_history.append(new_position)

        x, y = new_position
        value_new_position = obs['coords'][x][y]

        # отмечаем посещение клетки в файле сценария, кроме исследователя
        # if self.explorator:# or value_new_position[1] != ObjectStatus.plant.value:
        #     obs['coords'][x][y][0] = PointStatus.visited.value

        new_position, reward = self.get_agent_rewards(new_position, value_new_position[1])
        self.position = new_position

        info = {"done": int(sum(element[2] == Done.done.value for row
                                in self.env.current_map for element in row))}
        logging.info(f"{self.name} действие: {action} - позиция: {new_position}")

        return new_position, reward, terminated, truncated, info

    def get_observation(self) -> dict[str, np.array]:
        """
        Get observation of current agent, using view range of agent.
        :return: observation dictionary
        """
        coords = np.full((self.env.grid_size, self.env.grid_size, 3), fill_value=0)
        for pos in self.get_review():
            x, y = pos
            coords[x][y][0] = Point.viewed.value
            if pos in self.env.obstacle_positions:
                coords[x][y][1] = Obj.obstacle.value
            elif pos in self.env.plants_positions:
                coords[x][y][1] = Obj.plant.value

        observation = {
            'pos': self.position,
            'coords': coords
        }
        return observation

    def __repr__(self):
        return f'{self.name}'

    def get_agent_rewards(self, new_position: tuple[int, int], value: float) -> tuple[
            tuple[int, int], int]:
        """
        Update position of agent in dependency of cells.
        Give reward in dependency of cells.
        :param value: status and objects in agent position
        :param new_position: coordinates of agent (x, y)
        :return: coordinates of agent (x, y) and agent reward
        """
        agent_reward = 0
        # TEST вместо этих штрафов попробовать штраф за каждый шаг минимальный
        # TEST награда за удаление друг от друга
        if len(self.position_history) > 3:
            agent_reward += self.check_loop(new_position)

        if not ((self.env.margin <= new_position[0] <= self.env.inner_grid_size) and (
                self.env.margin <= new_position[1] <= self.env.inner_grid_size)):
            agent_reward -= c.PENALTY_OUT_FIELD
            logging.warning(f"{self} вышел за границы внутреннего поля: {new_position}")
            new_position = self.position
        elif value == Obj.obstacle.value:
            agent_reward -= c.PENALTY_OBSTACLE
            new_position = self.position
            logging.info(
                f"Упс, препятствие! {self} - штраф {c.PENALTY_OBSTACLE}, вернулся на {new_position}")

        return new_position, agent_reward

    def check_loop(self, new_position) -> int:
        """
        Calculate penalty for loop for agent position
        :param new_position: coordinates of agent (x, y)
        :return: penalty for loop
        """
        reward = 0
        pos_counter = self.position_history.count(new_position)
        if new_position == self.position:
            reward -= c.PENALTY_LOOP
            logging.warning(f"Штраф {self} за второй раз в одну клетку' {self.position}")
        elif 4 >= pos_counter > 2:
            reward -= c.PENALTY_LOOP * c.CORR_COEF
            logging.warning(
                f"Штраф {self} за вторичное посещение {new_position}"
                f" в последние {len(self.position_history)} шагов")
        elif pos_counter > 4:
            reward -= c.PENALTY_LOOP * c.CORR_COEF * c.CORR_COEF
            logging.warning(
                f"Штраф {self} за мнократное посещение {new_position}"
                f" в последние {len(self.position_history)} шагов")
        return reward

    def get_review(self) -> list[tuple[int, int]]:
        """
        Get cells in view range of current agent.
        :return: list of cell's coordination
        """
        review = []
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                x, y = self.position[0] + dx, self.position[1] + dy
                if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                    review.append((x, y))
        return review
