import torch as th
# Параметры модели
TIME = 30000
LEARNING_RATE = 0.0001
GAMMA = 0.99
CLIP_RANGE = 0.2
N_STEPS = 2048#4096 # должна быть привязка к кол-ву шагов эпизода
COEF = 0.001
VF_COEF = 0.5
CLIP_RANGE_VF = 0.2
N_EPOCHS = 5#0#0#0
BATCH_SIZE = 256#128
# policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))

# Параметры экрана и сетки
SCREEN_SIZE = 900
BAR_HEIGHT = SCREEN_SIZE * 0.13
GRID_SIZE = 20
MARGIN_SIZE = 1

# Параметры игры
NUM_AGENTS = 3
TARGET_PERCENT = 0.1#45
OBSTACLE_PERCENT = 0.05
STATION_SIZE = 2
MAX_STEPS_GAME = (GRID_SIZE ** 2) * 10 #add to class
VIEW_RANGE = 1  # Область зрения 3x3
ON_TARGET_CONSUMPTION = 10  # Расход
TANK_CAPACITY = 20 * ON_TARGET_CONSUMPTION  # Как-то продумать его
ENERGY_CAPACITY = 5000  # Максимальный запас энергии
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_DONE = 2
COUNT_ACTIONS = 9
MIN_GAME_STEPS = (GRID_SIZE ** 2 // NUM_AGENTS) * 8# add to class

# Награды
REWARD_EXPLORE = 1  # Вознаграждение за исследование новых клеток
REWARD_DONE = REWARD_EXPLORE * 1.1
REWARD_COMPLETION = REWARD_DONE * 100
PENALTY_LOOP = 0.1
PENALTY_RETURN = 0.05
PENALTY_OUT_FIELD = 0.1
PENALTY_OBSTACLE = 0.15
PENALTY_CRASH = 0.2

# Позиции цветов и ям
# PLACEMENT_MODE = 'fixed'
PLACEMENT_MODE = 'random'

FIXED_TARGET_POSITIONS = [
    (2, 2), (2, 8), (4, 3), (4, 7), (6, 2),
    (6, 8), (8, 4), (8, 6), (3, 5), (7, 5)
]

FIXED_OBSTACLE_POSITIONS = [
    (1, 1), (1, 9), (3, 3), (7, 7), (9, 5)
]

# Цвета, шрифты
WHITE = (200, 200, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (255, 69, 0)
GRAY = (30, 30, 30)
LIGHT_GRAY = (100, 100, 100)
FONT = 'Arial'

# изображения
AGENT = "images/drone.png"
TARGET_SPRAY = "images/bad_plant.png"
DONE_TARGET_SPRAY = "images/healthy_plant.png"
DONE_TARGET_EXPLORE = "images/explored.png"
OBSTACLES = "./images/obstacles"
STATION = "images/robdocst.png"
FIELD = "images/field.png"
FIELD_BACKGROUND = "images/forest.jpg"
