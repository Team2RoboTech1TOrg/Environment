import logging
from abc import ABC

import numpy as np

import const
from FarmingScenario import FarmingScenario


class SprayingScenario(FarmingScenario, ABC):
    def __init__(self):
        super().__init__()
        self.step_reward = None
        self.known_obstacles = None
        self.known_targets = None

    def render(self, mode):
        pass

    def step(self, actions):
        self.step_reward = 0
        for i, agent in enumerate(self.env.agents):
            new_position, agent_reward, terminated, truncated, info = agent.take_action(actions[i])
            if self.env.step_count != 1:
                new_position = self.env.check_crash(self.env.obs, agent, new_position)
            if self.env.obs['coords'][new_position[0]][new_position[1]] == 0:
                self.step_reward += const.REWARD_EXPLORE
                logging.info(f"{agent.name} исследовал новую клетку {new_position}")
                self.env.obs['coords'][new_position[0]][new_position[1]] = 1
            self.env.obs['pos'][i] = new_position
            self.step_reward += agent_reward

        # TO DO подумать, может как-то иначе реализовать без списков
        self.known_targets = np.argwhere(self.env.obs['coords'] == 4)
        self.known_obstacles = np.argwhere(self.env.obs['coords'] == 3)
        self.viewed_cells = np.argwhere(self.env.obs['coords'] == 1)

        reward, terminated, truncated, info = self.env._check_termination_conditions()
        self.env.step_count += 1
        logging.info(
            # f"Награда: {self.total_reward}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )

        return self.env.obs, reward, terminated, truncated, {}

    def reset(self):
        self.known_obstacles = set()
        self.known_targets = set()
        self.reset_objects_positions()
        self.step_reward = 0

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        unavailable_positions = {self.env.base_position}
        self.target_positions = self._get_objects_positions(unavailable_positions, const.COUNT_TARGETS)
        unavailable_positions.update(self.target_positions)
        self.obstacle_positions = self._get_objects_positions(unavailable_positions, const.COUNT_OBSTACLES)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = const.FIXED_TARGET_POSITIONS.copy()
        self.obstacle_positions = const.FIXED_OBSTACLE_POSITIONS.copy()
