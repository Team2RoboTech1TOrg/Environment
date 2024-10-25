from utils import load_image

# Параметры экрана и сетки
SCREEN_SIZE = 700
GRID_SIZE = 11
CELL_SIZE = SCREEN_SIZE // GRID_SIZE

#Game params
BASE_COORD = 5
COUNT_FLOWERS = 10
COUNT_HOLES = 5

# Параметры робота
VIEW_RANGE = 1  # Область зрения 3x3
WATER_CAPACITY = 100  # Максимальный запас воды
ENERGY_CAPACITY = 200000  # Максимальный запас энергии
WATER_CONSUMPTION = 10  # Расход воды на полив
ENERGY_CONSUMPTION = 1000  # Расход энергии на действие
COLLISION_PENALTY = -5  # Штраф за столкновение с ямой
BASE_RECHARGE_AMOUNT = 50000  # Восстановление энергии на базе
COUNT_ACTIONS = 6

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (132, 184, 56)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (169, 169, 169)

# Подгружаем изображения
AGENT_ICON = load_image("images/unit.png", CELL_SIZE)  # Изображение робота
FLOWER_ICON = load_image("images/clumb2.png", CELL_SIZE)  # Сухие цветы
WATERED_FLOWER_ICON = load_image("images/clumb1.png", CELL_SIZE)  # Политые цветы
HOLE_ICON = load_image("images/pit.png", CELL_SIZE)  # Яма
BASE_ICON = load_image("images/robdocst.png", CELL_SIZE)  # База