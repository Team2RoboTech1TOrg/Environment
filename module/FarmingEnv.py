import time
import pygame
import gymnasium as gym
import numpy as np

from Agent import Agent
from SystemObservationSpace import SystemObservationSpace
from logger import logging
import const
from utils import convert_to_multidiscrete, load_image, load_obstacles


class FarmingEnv(gym.Env):
    def __init__(self, scenario):
        super(FarmingEnv, self).__init__()
        self.scenario = scenario
        self.start_time = None
        self.total_reward = None
        self.step_reward = None
        self.agents = self.scenario.agents
        self.num_agents = self.scenario.num_agents
        self.grid_size = self.scenario.grid_size
        # self.step_count = None
        action_spaces = gym.spaces.Dict({
            f'agent_{i}': agent.action_space
            for i, agent in enumerate(self.scenario.agents)
        })
        self.action_space = convert_to_multidiscrete(action_spaces)
        self.observation_space = SystemObservationSpace(self.agents, self.num_agents,
                                                        self.grid_size)

    def reset(self, *, seed=None, options=None):
        self.start_time = time.time()
        # self.total_reward = 0
        # self.step_reward = 0
        obs, info = self.scenario.reset()
        logging.info("Перезагрузка среды")
        return obs, info

    def get_observation(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells
        """
        obs = self.scenario.get_observation()
        return obs

    def step(self, actions):
        # logging.info(f"Шаг: {self.step_count}")
        obs, reward, terminated, truncated, info = self.scenario.step(actions)
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
