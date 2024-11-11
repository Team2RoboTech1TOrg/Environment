from utils import load_image

# Параметры модели
LEARNING_RATE = 0.0001
GAMMA = 0.99
CLIP_RANGE = 0.2
N_STEPS = 4096
COEF = 0.01
CLIP_RANGE_VF = 0.2
N_EPOCHS = 3
BATCH_SIZE = 64

# Параметры экрана и сетки
SCREEN_SIZE = 900
GRID_SIZE = 13
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
MARGIN_SIZE = 1

# Параметры игры
NUM_AGENTS = 3
BASE_COORD = GRID_SIZE // 2
COUNT_FLOWERS = 10
COUNT_HOLES = 5
MAX_STEPS_GAME = 3000
VIEW_RANGE = 1  # Область зрения 3x3
WATER_CONSUMPTION = 10  # Расход воды на полив
WATER_CAPACITY = COUNT_FLOWERS * WATER_CONSUMPTION  # Максимальный запас воды
ENERGY_CAPACITY = 2000  # Максимальный запас энергии
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_WATER = 2
COUNT_ACTIONS = 4
MIN_GAME_STEPS = GRID_SIZE * 10

# Награды
REWARD_COMPLETION = 500
REWARD_EXPLORE = 50  # Вознаграждение за исследование новых клеток
PENALTY_LOOP = 10
PENALTY_OUT_FIELD = 20
PENALTY_HOLE = 10

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
