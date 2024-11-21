# Параметры модели
TIME = 50_000
LEARNING_RATE = 0.0001
GAMMA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 4096
COEF = 0.001
VF_COEF = 0.4
CLIP_RANGE_VF = 0.2
N_EPOCHS = 30
BATCH_SIZE = 128

# Параметры экрана и сетки
SCREEN_SIZE = 900
BAR_HEIGHT = SCREEN_SIZE * 0.13
GRID_SIZE = 20
MARGIN_SIZE = 1

# Параметры игры
NUM_AGENTS = 3
# COUNT_TARGETS = ceil(GRID_SIZE ** 2 * 0.45)
# COUNT_OBSTACLES = ceil(GRID_SIZE ** 2 * 0.05)
TARGET_PERCENT = 0.45
OBSTACLE_PERCENT = 0.05
STATION_SIZE = 2
MAX_STEPS_GAME = (GRID_SIZE ** 2) * 5 #add to class
VIEW_RANGE = 1  # Область зрения 3x3
ON_TARGET_CONSUMPTION = 10  # Расход
TANK_CAPACITY = 20 * ON_TARGET_CONSUMPTION  # Как-то продумать его
ENERGY_CAPACITY = 3000  # Максимальный запас энергии
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_DONE = 2
COUNT_ACTIONS = 5
MIN_GAME_STEPS = (GRID_SIZE * GRID_SIZE // NUM_AGENTS) * 2 # add to class

# Награды
REWARD_EXPLORE = 2  # Вознаграждение за исследование новых клеток
REWARD_DONE = REWARD_EXPLORE * 1.5
REWARD_COMPLETION = REWARD_DONE * 10
PENALTY_LOOP = 1
PENALTY_OUT_FIELD = 1
PENALTY_OBSTACLE = 2
PENALTY_CRASH = 3

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
