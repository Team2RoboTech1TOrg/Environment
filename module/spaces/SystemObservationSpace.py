from typing import List

import gymnasium as gym
import numpy as np

from agent.Agent import Agent


class SystemObservationSpace(gym.spaces.Dict):
    def __init__(self, agents: List[Agent], num_agents: int, grid_size: int):
        self.positions_space = gym.spaces.Box(
            low=np.stack([agent.observation_space.position_space.low for agent in agents], axis=0),
            high=np.stack([agent.observation_space.position_space.high for agent in agents], axis=0),
            shape=(num_agents, 2),
            dtype=np.int32)

        self.coords_space = gym.spaces.Box(
            low=np.zeros((grid_size, grid_size, 3), dtype=np.int32),
            high=np.full((grid_size, grid_size, 3), fill_value=3, dtype=np.int32),
            dtype=np.int32
        )

        observation_space = {
            'pos': self.positions_space,
            'coords': self.coords_space
        }
        super().__init__(observation_space)
