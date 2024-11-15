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
        # self.step_count = None
        self.action_space = self.scenario.action_space
        self.observation_space = self.scenario.observation_space

    def reset(self, *, seed=None, options=None):
        self.start_time = time.time()
        self.total_reward = 0
        self.step_reward = 0
        obs = self.scenario.reset()
        logging.info("Перезагрузка среды")
        return obs, {}

    def get_observation(self):
        """
        Get observation at the moment: array of agents positions and
        current map with actual status of cells
        """
        obs = self.scenario.get_observation()
        return obs

    def step(self, actions):
        # logging.info(f"Шаг: {self.step_count}")
        obs, reward, terminated, truncated, info = self.scenario.step()
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render agent game"""
        return self.scenario.render()

    def render_message(self, render_text: str):
        """
        Display message in the center of screen
        :param render_text: str
        :return:
        """
        return self.scenario.render_message(render_text)

