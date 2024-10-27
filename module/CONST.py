from utils import load_image

# Параметры модели
LEARNING_RATE = 0.001
GAMMA = 0.98
CLIP_RANGE = 0.2
N_STEPS = 4096
COEF = 0.01

# Параметры экрана и сетки
SCREEN_SIZE = 700
GRID_SIZE = 11
CELL_SIZE = SCREEN_SIZE // GRID_SIZE

# Параметры игры
BASE_COORD = 5
COUNT_FLOWERS = 10
COUNT_HOLES = 5
VIEW_RANGE = 1  # Область зрения 3x3
WATER_CAPACITY = 50  # Максимальный запас воды
ENERGY_CAPACITY = 100000  # Максимальный запас энергии
WATER_CONSUMPTION = 10  # Расход воды на полив
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_WATER = 2
ENERGY_RECHARGE_AMOUNT = 100
WATER_REFILL_AMOUNT = 50
COUNT_ACTIONS = 6
MAX_STEPS_WITHOUT_PROGRESS = 1000
MAX_ENERGY_WITHOUT_PROGRESS = 800
MAX_HOLE_FALL = 5
MAX_TIME = 3000
MAX_STEPS_DISTANCE = 50
MAX_DISTANCE_FROM_FLORAL = 5
MIN_FLOWERS_TO_WATER = 5
REWARD_COMPLETION = 1000
REWARD_MOVE = -0.5
REWARD_WATER_SUCCESS = 200
REWARD_WATER_FAIL_ALREADY_WATERED = -5
REWARD_WATER_FAIL_NOT_ON_FLOWER = -1000
REWARD_RECHARGE = -10
REWARD_REFILL = -10
REWARD_COLLISION = -1000  # штраф за попадание в яму
REWARD_EXPLORE = 10  # Штраф за ненужное исследование
REWARD_AVOID_HOLE = 10  # Вознаграждение за обход ям
REWARD_TIME = lambda t: 1 / t if t > 0 else 0
REWARD_STEPS = lambda m: 1 / m if m > 0 else 0
PENALTY_LOOP = -10

# Позиции цветов и ям
# PLACEMENT_MODE = 'fixed'
PLACEMENT_MODE = 'random'

FIXED_FLOWER_POSITIONS = [
    (2, 2), (2, 8), (4, 3), (4, 7), (6, 2),
    (6, 8), (8, 4), (8, 6), (3, 5), (7, 5)
]

FIXED_HOLE_POSITIONS = [
    (1, 1), (1, 9), (3, 3), (7, 7), (9, 5)
]

# Цвета, шрифты
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (132, 184, 56)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (169, 169, 169)
FONT_SIZE = 24
TITLE_SIZE = 60

# Подгружаем изображения
AGENT_ICON = load_image("images/unit.png", CELL_SIZE)  # Изображение робота
FLOWER_ICON = load_image("images/clumb2.png", CELL_SIZE)  # Сухие цветы
WATERED_FLOWER_ICON = load_image("images/clumb1.png", CELL_SIZE)  # Политые цветы
HOLE_ICON = load_image("images/pit.png", CELL_SIZE)  # Яма
BASE_ICON = load_image("images/robdocst.png", CELL_SIZE)  # База
