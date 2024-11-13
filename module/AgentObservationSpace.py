import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AgentObservationSpace(gym.Space):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

        self.position_space = spaces.Box(
            low=-self.size,
            high=self.size,
            shape=(2,),
            dtype=np.int32
        )

        self.points_space = spaces.Box(
            low=0,
            high=5,
            shape=(self.size, self.size),
            dtype=np.int32
        )

        self.observation_space = spaces.Dict({
            'pos': self.position_space,
            'coords': self.points_space
        })

    def get_agent_positions(self):
        return self.position_space
