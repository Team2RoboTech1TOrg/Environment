import gymnasium as gym
from gymnasium import spaces

from WateringEnv import WateringEnv

#
# class CustomObservationSpace(gym.Space):
#     def __init__(self, env, points, point_types):
#         super().__init__()
#         self.env = env
#         self.points = points
#         self.point_types = point_types
#
#         self.agent_observations = spaces.Dict({
#             f'agent_{i}': agent.observation_space
#             for i, agent in enumerate(self.env.agents)
#         })
#         self.points_observations = spaces.Dict({
#             coord: spaces.Discrete(point_types) for coord in points.keys()
#         })
#
#         self.observation_space = spaces.Dict({
#             'agents': self.agent_observations,
#             'points': self.points_observations
#         })


# env = WateringEnv(2)
# obs_space = CustomObservationSpace(env, {3:2}, [3])
# discovered_points = {(3, 2): 3}
# point_types = {(3, 2)}
# print(spaces.Dict({
#     coord: spaces.Discrete(len(point_types[coord])) for coord in discovered_points.keys()
# }))
