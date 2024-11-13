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

    # self.observation_space = gym.spaces.Dict(
    #     spaces={
    #         "position": gym.spaces.Box(low=0, high=(self.grid_size - 1), shape=(2,), dtype=np.int32),
    #         "direction": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
    #         "grid": gym.spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.uint8),
    #     })

    # self.observation_space = spaces.Dict({
    #         'agents': spaces.Dict({aid: spaces.Box(low=0, high=max_x, shape=(2,), dtype=np.int32) for aid in self.agents}),
    #         'cells': spaces.Dict({(x, y): spaces.Discrete(2) for x in range(max_x) for y in range(max_y)})
    #     })