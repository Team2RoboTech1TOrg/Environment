from utils import load_image

# Параметры модели
LEARNING_RATE = 0.0001
GAMMA = 0.99
CLIP_RANGE = 0.2
N_STEPS = 4096
COEF = 0.01
CLIP_RANGE_VF = 0.2
N_EPOCHS = 150
BATCH_SIZE = 64

# Параметры экрана и сетки
SCREEN_SIZE = 700
GRID_SIZE = 11
CELL_SIZE = SCREEN_SIZE // GRID_SIZE

# Параметры игры
BASE_COORD = 5
COUNT_FLOWERS = 10
COUNT_HOLES = 5
MAX_STEPS_GAME = 10000
VIEW_RANGE = 1  # Область зрения 3x3
WATER_CAPACITY = 300  # Максимальный запас воды
ENERGY_CAPACITY = 5000  # Максимальный запас энергии
WATER_CONSUMPTION = 10  # Расход воды на полив
ENERGY_CONSUMPTION_MOVE = 1
ENERGY_CONSUMPTION_WATER = 2
# ENERGY_RECHARGE_AMOUNT = 100
# WATER_REFILL_AMOUNT = 50
COUNT_ACTIONS = 5
# MAX_STEPS_WITHOUT_PROGRESS = 1000
# MAX_ENERGY_WITHOUT_PROGRESS = 800
# MAX_TIME = 3000
MAX_STEPS_DISTANCE = 5
MAX_DISTANCE_FROM_FLORAL = 10
MIN_FLOWERS_TO_WATER = 5
MIN_GAME_STEPS = 2000

# Награды
# REWARD_APPROACH_UNWATERED_FLOWER = 2  # Увеличенное вознаграждение за приближение к известному неполитому цветку
REWARD_WATER_KNOWN_FLOWER = 1000  # Дополнительное вознаграждение за полив неполитого известного цветка
REWARD_COMPLETION = 10000
REWARD_WATER_SUCCESS = 800 # Дополнительное вознаграждение за полив неполитого известного цветка
PENALTY_WATER_FAIL_ALREADY_WATERED = -10  # Если цветок уже полит
PENALTY_WATER_FAIL_NOT_ON_FLOWER = -15  # Агент попытался полить не находясь на цветке
NEXT_2_UNWATERED_FLOWER = 5  # Вознаграждение за нахождение рядом с неполитым цветком
REWARD_MAX_STEPS_DISTANCE = -10
# REWARD_UNNECESSARY_MOVE = -15  # Штраф за ненужное движение
# REWARD_REFILL = -10
# REWARD_BASE_BACK = 5  # На базу с низким зарядом
# PENALTY_BASE_BACK = -15  # На базу с большим зарядом
PENALTY_COLLISION = -15  # штраф за попадание в яму
REWARD_EXPLORE = 5  # Вознаграждение за исследование новых клеток
DONT_WATERING = - 5
# REWARD_AVOID_HOLE = 5 # Вознаграждение за обход ям
# REWARD_TIME = lambda t: 1 / t if t > 0 else 0
# REWARD_STEPS = lambda m: 1 / m if m > 0 else 0
# PENALTY_LOW_ENERGY_NO_PROGRESS = -10  # Низкий уровень энергии без прогресса
PENALTY_LOOP = -15

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
