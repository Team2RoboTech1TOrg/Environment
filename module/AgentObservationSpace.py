import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AgentObservationSpace(gym.spaces.Dict):
    def __init__(self, size: int):
        self.position_space = spaces.Box(
            low=0,
            high=size,
            shape=(2,),
            dtype=np.int32
        )

        self.points_space = spaces.Box(
            low=np.zeros((size, size, 2), dtype=np.int32),  # new
            high=np.full((size, size, 2), fill_value=2, dtype=np.int32),  # new
            # shape=(size, size),
            dtype=np.int32
        )

        observation_space = {
            'pos': self.position_space,
            'coords': self.points_space
        }
        super().__init__(observation_space)

    def get_agent_positions(self):
        return self.position_space
