import gymnasium as gym
import numpy as np

from Agent import Agent


class SystemObservationSpace(gym.Space):
    def __init__(self, agents: list[Agent], num_agents: int, grid_size: int):
        super().__init__()
        self.size = grid_size
        self.agents = agents
        self.num_agents = num_agents

        self.position_space = gym.spaces.Box(
            low=np.stack([agent.observation_space.position_space.low for agent in self.agents], axis=0),
            high=np.stack([agent.observation_space.position_space.high for agent in self.agents], axis=0),
            shape=(self.num_agents, 2),
            dtype=np.int32)

        self.points_space = gym.spaces.Box(low=0, high=5, shape=(self.size, self.size), dtype=np.int32)

        self.observation_space = gym.spaces.Dict({
            'pos': self.position_space,
            'coords': self.points_space
        })
