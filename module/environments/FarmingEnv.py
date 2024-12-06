import gymnasium as gym
from gymnasium.core import ActType

import const
from spaces.SystemObservationSpace import SystemObservationSpace
from logging_system.logger import logging


class FarmingEnv(gym.Env):
    def __init__(self, scenario):
        super(FarmingEnv, self).__init__()
        self.scenario = scenario
        self.agents = self.scenario.agents
        self.num_agents = self.scenario.num_agents
        self.grid_size = self.scenario.grid_size
        self.action_space = gym.spaces.Discrete(const.COUNT_ACTIONS)
        self.observation_space = SystemObservationSpace(self.agents, self.num_agents,
                                                        self.grid_size)

    def reset(self, *, seed=None, options=None):
        logging.info("Перезагрузка среды")
        obs, info = self.scenario.reset()
        return obs, info

    def get_observation(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells
        """
        obs = self.scenario.get_observation()
        return obs

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = self.scenario.step(action)
        # print(reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render agent game"""
        return self.scenario.render()

    def render_message(self, render_text: str):
        """        Display message in the center of screen
        :param render_text: str
        :return:
        """
        return self.scenario.render_message(render_text)
