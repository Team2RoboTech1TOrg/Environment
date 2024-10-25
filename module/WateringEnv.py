import time
import pygame
import gymnasium as gym
import numpy as np

from CONST import VIEW_RANGE, ENERGY_CAPACITY, WATER_CAPACITY, WATER_CONSUMPTION, GRID_SIZE, COUNT_ACTIONS, BASE_ICON, \
    CELL_SIZE, BASE_COORD, GREEN, AGENT_ICON, SCREEN_SIZE, BLACK, WHITE, HOLE_ICON, FLOWER_ICON, WATERED_FLOWER_ICON, \
    COUNT_HOLES, COUNT_FLOWERS


class WateringEnv(gym.Env):
    def __init__(self):
        super(WateringEnv, self).__init__()

        self.grid_size = GRID_SIZE
        self.action_space = gym.spaces.Discrete(COUNT_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size, shape=(3, ),#, + 2 * VIEW_RANGE ** 2,),
                                                dtype=np.int32)  # Пространство наблюдений

        # Определение недоступных позиций
        unavailable_positions = {BASE_COORD * self.grid_size + BASE_COORD}  # База

        # Случайное размещение 10 цветов
        available_positions = [i for i in range(self.grid_size ** 2) if
                               i not in unavailable_positions]
        targets = np.random.choice(available_positions, size=COUNT_FLOWERS,
                                   replace=False)
        unavailable_positions.update(targets)

        # Случайное размещение 5 ям
        available_positions = [i for i in range(self.grid_size ** 2) if
                               i not in unavailable_positions]
        holes = np.random.choice(available_positions, size=COUNT_HOLES,
                                 replace=False)

        # Преобразование позиций в координаты
        self.target_positions = np.array([[pos // self.grid_size, pos % self.grid_size] for pos in
                                          targets])
        self.hole_positions = np.array([[pos // self.grid_size, pos % self.grid_size] for pos in
                                        holes])

        # Инициализация начальных параметров робота
        self.agent_position = tuple([BASE_COORD, BASE_COORD])  # Стартовая позиция на базе
        self.watered_status = np.zeros(10)  # Статус всех цветов (0 - не полит, 1 - полит)
        self.water_tank = WATER_CAPACITY  # Объем воды в баке
        self.energy = ENERGY_CAPACITY  # Уровень энергии
        self.sensors = {}  # Сенсоры робота (сейчас пустые)
        self.start_time = time.time()  # Начальное время
        self.score = 0  # Счёт игры

    def reset(self, *, seed=None, options=None):
        # Сброс состояния среды
        self.agent_position = tuple([BASE_COORD, BASE_COORD])  # Возвращаем робота на базу
        self.watered_status = np.zeros(10)  # Все цветы становятся не политыми
        self.water_tank = WATER_CAPACITY  # Заполняем бак водой
        self.energy = ENERGY_CAPACITY  # Полный заряд энергии
        self.sensors = {}  # Обнуляем данные сенсоров
        self.start_time = time.time()  # Обновляем начальное время
        self.score = 0  # Сброс очков
        return self.get_observation(), {}

    def get_observation(self):
        # Формирование области, видимой агентом
        visible_area = []
        for dx in range(-VIEW_RANGE, VIEW_RANGE + 1):  # Проход по соседним клеткам в радиусе 1
            for dy in range(-VIEW_RANGE, VIEW_RANGE + 1):
                x, y = np.array(self.agent_position) + np.array([dx, dy])  # Рассчитываем координаты
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:  # Проверка на выход за пределы сетки
                    visible_area.append((x, y))  # Добавляем видимую клетку в список
        return np.array(
            [self.agent_position[0], self.agent_position[1], self.water_tank])# + [coord for pair in visible_area for
                                                                               #  coord in
                                                                                # pair])  # Возвращаем наблюдение

    # Метод шага среды (обработка действия)
    def step(self, action):
        # Если энергия исчерпана, агент не может двигаться
        if self.energy <= 0:
            return self.get_observation(), -100, True, True, {}

        # Действия агента в зависимости от выбранного действия
        if action == 0:  # Движение вверх
            self.agent_position = (max(0, self.agent_position[0] - 1), self.agent_position[1])
        elif action == 1:  # Движение вниз
            self.agent_position = (min(self.grid_size - 1, self.agent_position[0] + 1), self.agent_position[1])
        elif action == 2:  # Движение влево
            self.agent_position = (self.agent_position[0], max(0, self.agent_position[1] - 1))
        elif action == 3:  # Движение вправо
            self.agent_position = (self.agent_position[0], min(self.grid_size - 1, self.agent_position[1] + 1))
        elif action == 4:  # Полив цветка
            for i, pos in enumerate(self.target_positions):
                if np.array_equal(pos, self.agent_position):  # Проверяем, находится ли агент на позиции цветка
                    if self.watered_status[i] == 0 and self.water_tank > 0:  # Если цветок не полит и есть вода
                        self.watered_status[i] = 1  # Поливаем цветок
                        self.water_tank -= WATER_CONSUMPTION  # Уменьшаем запас воды
                        self.score += 50  # Добавляем очки за полив
                        reward = 50  # Награда за полив
                    else:
                        reward = -1  # Штраф, если цветок уже полит или нет воды
                    break
        elif action == 5:  # Зарядка на базе
            if self.agent_position == (BASE_COORD, BASE_COORD):  # Проверяем, находится ли агент на базе
                self.energy = min(self.energy + 50000, ENERGY_CAPACITY)  # Восстанавливаем энергию

        # Проверяем, попал ли агент в яму
        if any(np.array_equal(hole, self.agent_position) for hole in self.hole_positions):
            self.energy -= 60  # Уменьшаем энергию за попадание в яму
            self.water_tank = max(0, self.water_tank - 20)  # Уменьшаем воду, но не менее 0

        # Проверяем, все ли цветы политы
        done = np.all(self.watered_status == 1) and self.agent_position == (
            5, 5)  # Если все цветы политы и агент вернулся на базу, эпизод завершён
        reward = -1  # Штраф за каждый шаг

        if done:
            reward = 1000  # Награда за завершение миссии
            self.score += reward
        truncated = False
        # Return the observation, reward, done, truncated, and info
        return self.get_observation(), reward, done, truncated, {}  # Возврат наблюдения, награды и состояния эпизода

    def render(self, screen):
        # Отрисовка среды
        screen.fill(GREEN)  # Заливаем фон зеленым цветом, представляющим траву

        # Отрисовка сетки
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                                 1)  # Рисуем черную границу вокруг каждой клетки

        # Отрисовка базы
        screen.blit(BASE_ICON, (5 * CELL_SIZE, 5 * CELL_SIZE))  # Отображаем изображение базы в центре сетки

        # Отрисовка целей (цветы) и ям
        for i, pos in enumerate(self.target_positions):  # Проход по всем цветам
            icon = WATERED_FLOWER_ICON if self.watered_status[
                i] else FLOWER_ICON  # Если цветок полит, отображаем политый, иначе сухой
            screen.blit(icon, (pos[1] * CELL_SIZE, pos[0] * CELL_SIZE))  # Отображаем цветок на его позиции

        for hole in self.hole_positions:  # Проход по всем ямам
            screen.blit(HOLE_ICON, (hole[1] * CELL_SIZE, hole[0] * CELL_SIZE))  # Отображаем яму на её позиции

        # Отрисовка времени, очков, заряда и уровня воды
        elapsed_time = time.time() - self.start_time  # Рассчитываем прошедшее время
        font = pygame.font.SysFont(None, 36)  # Устанавливаем шрифт для текста
        time_text = font.render(f"Time: {elapsed_time:.2f} sec", True, BLACK)  # Отображение времени
        score_text = font.render(f"Score: {self.score}", True, BLACK)  # Отображение очков
        energy_text = font.render(f"Energy: {self.energy}", True, BLACK)  # Отображение уровня энергии
        water_text = font.render(f"Water: {self.water_tank}", True, BLACK)  # Отображение уровня воды

        status_bar_height = 100  # Высота панели статуса
        status_bar_rect = pygame.Rect(0, SCREEN_SIZE, SCREEN_SIZE + 100,
                                      status_bar_height)  # Прямоугольник для панели статуса
        pygame.draw.rect(screen, WHITE, status_bar_rect)  # Рисуем белый прямоугольник для панели статуса

        screen.blit(time_text, (10, SCREEN_SIZE + 20))  # Отображение времени на панели
        screen.blit(score_text, (10, SCREEN_SIZE + 60))  # Отображение очков на панели
        screen.blit(energy_text, (200, SCREEN_SIZE + 20))  # Отображение энергии на панели
        screen.blit(water_text, (200, SCREEN_SIZE + 60))  # Отображение уровня воды на панели

        # Отрисовка агента поверх остальных объектов
        screen.blit(AGENT_ICON, (self.agent_position[1] * CELL_SIZE,
                                 self.agent_position[0] * CELL_SIZE))  # Отображение агента в его текущей позиции

        pygame.display.flip()  # Обновление экрана
