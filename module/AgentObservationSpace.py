import gymnasium as gym
import numpy as np
from gymnasium import spaces


# class AgentObservationSpace(gym.spaces.Dict):
    # def __init__(self, size: int):
        # self.position_space = spaces.Box(
            # low=0,
            # high=size,
            # shape=(2,),
            # dtype=np.int32
        # )

        # self.points_space = spaces.Box(
            # low=np.zeros((size, size, 3), dtype=np.int32),
            # high=np.full((size, size, 3), fill_value=2, dtype=np.int32),
            # dtype=np.int32
        # )

        # observation_space = {
            # 'pos': self.position_space,
            # 'coords': self.points_space
        # }
        # super().__init__(observation_space)

class AgentObservationSpace(gym.spaces.Dict):
    def __init__(self, grid_size):
        self.position_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([grid_size - 1, grid_size - 1], dtype=np.int32),
            dtype=np.int32
        )
        self.coords_space = gym.spaces.Box(
            low=np.zeros((grid_size, grid_size, 3), dtype=np.int32),
            high=np.full((grid_size, grid_size, 3), fill_value=2, dtype=np.int32),
            dtype=np.int32
        )
        observation_space = {
            'pos': self.position_space,
            'coords': self.coords_space
        }

    def get_agent_positions(self):
        return self.position_space
