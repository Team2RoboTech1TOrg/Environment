import gymnasium as gym
import numpy as np
from gymnasium import spaces

from WateringEnv import WateringEnv


class CustomObservationSpace(gym.Space):
    def __init__(self, env, discovered_points, point_types):
        super().__init__()
        self.env = env
        self.discovered_points = discovered_points
        self.point_types = point_types

        # Формируем пространство наблюдений для каждого агента
        self.agent_observations = spaces.Dict({
            f'agent_{i}': agent.observation_space
            for i, agent in enumerate(self.env.agents)
        })

        # Формируем пространство наблюдений для обнаруженных точек
        self.points_observations = spaces.Dict({
            coord: spaces.Discrete(len(point_types[coord])) for coord in discovered_points.keys()
        })

        # Общий словарь пространства наблюдений
        self.observation_space = spaces.Dict({
            'agents': self.agent_observations,
            'points': self.points_observations
        })


# env = WateringEnv(2)
# obs_space = CustomObservationSpace(env, {3:2}, [3])
discovered_points = {(3, 2): 3}
point_types = {(3, 2)}
print(spaces.Dict({
    coord: spaces.Discrete(len(point_types[coord])) for coord in discovered_points.keys()
}))
