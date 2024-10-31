import time
from collections import deque

import pygame
import gymnasium as gym
import numpy as np

from logger import logging
import const


class WateringEnv(gym.Env):
    def __init__(self):
        super(WateringEnv, self).__init__()
        self.grid_size = const.GRID_SIZE
        self.screen = pygame.display.set_mode((const.SCREEN_SIZE, const.SCREEN_SIZE + 120))

        self.base_position = (const.BASE_COORD, const.BASE_COORD)
        self.agent_position = None  # Стартовая позиция
        self.water_tank = None  # Заполняем бак водой
        self.energy = None  # Полный заряд энергии
        self.start_time = None  # Начальное время
        # self.score = None
        self.reward = None
        self.watered_status = None  # Статус всех цветов (0 - не полит, 1 - полит)
        self.visited = None
        self.step_count = None
        self.hole_fall_count = None
        self.last_progress_step = None
        self.steps_since_last_distance = None
        self.distance_progress_steps = None
        self.position_history = None
        self.known_holes = None
        self.known_flowers = None
        self.explored_cells = None
        self.prev_distance_to_flower = None
        self.prev_distance_to_hole = None

        self.action_space = gym.spaces.Discrete(const.COUNT_ACTIONS)
        self.action_history = deque(maxlen=5)
        self.observation_space = gym.spaces.Box(
            low=-self.grid_size,  # Позволяет delta_x и delta_y быть отрицательными
            high=self.grid_size,
            shape=(35,),  # Надо описать магию
            dtype=np.float32
        )

    def reset_objects_positions(self):
        """
        Reset positions of objects
        :return: function for get object's postitions
        """
        if const.PLACEMENT_MODE == 'random':
            self._randomize_positions()
        elif const.PLACEMENT_MODE == 'fixed':
            self._fixed_positions()
        else:
            raise ValueError("Invalid PLACEMENT_MODE. Choose 'random' or 'fixed'.")

    def _randomize_positions(self):
        """
        Get random positions of objects
        """
        unavailable_positions = {self.base_position}
        self.target_positions = self._get_objects_positions(unavailable_positions, const.COUNT_FLOWERS)
        unavailable_positions.update(self.target_positions)
        self.hole_positions = self._get_objects_positions(unavailable_positions, const.COUNT_HOLES)

    def _fixed_positions(self):
        """
        Get fixed positions of objects
        """
        self.target_positions = const.FIXED_FLOWER_POSITIONS.copy()
        self.hole_positions = const.FIXED_HOLE_POSITIONS.copy()

    def _get_available_positions(self, unavailable: set) -> list:
        """
        Function for get available positions from all positions - unavailable
        :param unavailable: set
        :return: available positions
        """
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        return [pos for pos in all_positions if pos not in unavailable]

    def _get_objects_positions(self, unavailable: (), size: int) -> list:
        """
        Get list of object's positions using unavailable positions
        :param unavailable: set()
        :param size: int
        :return: list of positions [x, y]
        """
        available_positions = self._get_available_positions(unavailable)
        indices = np.random.choice(len(available_positions), size=size, replace=False)
        return [available_positions[i] for i in indices]

    def reset(self, *, seed=None, options=None):
        self.reset_objects_positions()
        self.agent_position = self.base_position
        self.watered_status = np.zeros(const.COUNT_FLOWERS)
        self.water_tank = const.WATER_CAPACITY
        self.energy = const.ENERGY_CAPACITY
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.visited[self.agent_position] = 1
        self.start_time = time.time()  # time.perf_counter()
        # self.score = 0
        self.reward = 0
        self.step_count = 0
        self.hole_fall_count = 0
        self.last_progress_step = 0
        self.steps_since_last_distance = 0
        self.distance_progress_steps = 0
        self.position_history = {}
        self.action_history.clear()
        self.known_holes = set()
        self.known_flowers = set()
        self.explored_cells = set()
        self.prev_distance_to_flower = -1
        self.prev_distance_to_hole = -1
        logging.info("Environment reset")
        obs = self._get_observation()
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape {obs.shape} does not match observation_space {self.observation_space.shape}"
            )
        return obs, {}

    def _get_observation(self):
        visible_area = []
        for dx in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
            for dy in range(-const.VIEW_RANGE, const.VIEW_RANGE + 1):
                x, y = self.agent_position[0] + dx, self.agent_position[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    pos = (x, y)
                    if pos in self.hole_positions:
                        visible_area.extend([1, 0])  # Яма
                        if pos not in self.known_holes:
                            self.known_holes.add(pos)
                            logging.debug(f"Новая известная яма: {pos}")
                    elif pos in self.target_positions:
                        idx = self.target_positions.index(pos)
                        watered = self.watered_status[idx]
                        visible_area.extend([2, watered])  # Цветок
                        if pos not in self.known_flowers:
                            self.known_flowers.add(pos)
                            logging.debug(f"Новый известный цветок: {pos}")
                    else:
                        visible_area.extend([0, 0])  # Пустая клетка
                    self.explored_cells.add(pos)
                else:
                    visible_area.extend([-1, 0])  # Вне границ

        # Вычисляем расстояние до ближайшего известного, но неполитого цветка
        known_unwatered_flowers = [
            pos for idx, pos in enumerate(self.target_positions)
            if self.watered_status[idx] == 0 and pos in self.known_flowers
        ]
        if known_unwatered_flowers:
            nearest_flower = min(
                known_unwatered_flowers,
                key=lambda pos: abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
            )
            distance_to_flower = abs(self.agent_position[0] - nearest_flower[0]) + abs(
                self.agent_position[1] - nearest_flower[1])
            delta_x = nearest_flower[0] - self.agent_position[0]
            delta_y = nearest_flower[1] - self.agent_position[1]
        else:
            distance_to_flower = -1  # Нет известных неполитых цветов
            delta_x = 0
            delta_y = 0

        # Вычисляем расстояние до ближайшей известной ямы
        if self.known_holes:
            distance_to_hole = min([
                abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
                for pos in self.known_holes
            ])
        else:
            distance_to_hole = -1  # Нет известных ям

        is_on_base = 1 if self.agent_position == self.base_position else 0
        is_on_flower = 1 if self.agent_position in self.target_positions else 0
        is_on_hole = 1 if self.agent_position in self.hole_positions else 0
        remaining_flowers = const.COUNT_FLOWERS - np.sum(self.watered_status)
        action_hist = list(self.action_history) + [0] * (5 - len(self.action_history))

        observation = np.concatenate([
            np.array(self.agent_position, dtype=float),  # 0-1
            np.array([self.water_tank, self.energy], dtype=float),  # 2-3
            np.array(visible_area, dtype=float),  # 4-21
            np.array([distance_to_flower, distance_to_hole], dtype=float),  # 22-23
            np.array([is_on_base, is_on_flower, is_on_hole, remaining_flowers], dtype=float),  # 24-27
            np.array([delta_x, delta_y], dtype=float),  # 28-29
            np.array(action_hist, dtype=float)  # 30-34
        ])  # Всего 35 элемента
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape {observation.shape} does not match observation_space {self.observation_space.shape}"
            )
        return observation

    def step(self, action):
        self.step_count += 1

        if self.energy < 10:
            return self._get_observation(), -100, True, False, {}

        if self.water_tank <= 10:  # возврат на базу за водой
            self.agent_position = self.base_position
            self.water_tank = const.WATER_CAPACITY

        # Запись истории позиций для обнаружения циклов
        pos_key = self.agent_position
        self.position_history[pos_key] = self.position_history.get(pos_key, 0) + 1
        if self.position_history[pos_key] > 5:
            # print(self.action_history, pos_key)
            self.reward += const.PENALTY_LOOP
        self.action_history.append(action)

        # Сохраняем предыдущие расстояния
        prev_distance_to_flower = self.prev_distance_to_flower
        prev_distance_to_hole = self.prev_distance_to_hole

        # Обновляем наблюдение, чтобы получить новые расстояния
        obs = self._get_observation()
        distance_to_flower = obs[22]
        distance_to_hole = obs[23]
        delta_x = obs[28]
        delta_y = obs[29]

        # Определяем известные неполитые цветки
        known_unwatered_flowers = [
            pos for idx, pos in enumerate(self.target_positions)
            if self.watered_status[idx] == 0 and pos in self.known_flowers
        ]

        prev_position =([obs[0], obs[1]])
        # idx = self.target_positions.index(prev_position)
        if (prev_position in self.target_positions) and (action != 5) and self.watered_status[[obs[0], obs[1]]] == 0:
            self.reward -= 5
            logging.info(f"Должен был поливать цветок на {prev_position}, но выбрал иное действие {self.agent_position}")

        # Проверяем, уменьшилось ли расстояние до ближайшего известного, но неполитого цветка
        if (prev_distance_to_flower != -1 and distance_to_flower != -1
                and distance_to_flower < prev_distance_to_flower):
            self.reward += const.REWARD_APPROACH_UNWATERED_FLOWER
            logging.info(f"Уменьшилось расстояние до неполитого цветка {self.agent_position}")

        # Обновляем сохраненные расстояния
        self.prev_distance_to_flower = distance_to_flower
        self.prev_distance_to_hole = distance_to_hole

        # Действия агента в зависимости от выбранного действия
        match action:
            case 0:  # Вверх
                new_position = (max(0, self.agent_position[0] - 1), self.agent_position[1])
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 1:  # Вниз
                new_position = (min(self.grid_size - 1, self.agent_position[0] + 1), self.agent_position[1])
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 2:  # Влево
                new_position = (self.agent_position[0], max(0, self.agent_position[1] - 1))
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 3:  # Вправо
                new_position = (self.agent_position[0], min(self.grid_size - 1, self.agent_position[1] + 1))
                self.energy -= const.ENERGY_CONSUMPTION_MOVE
            case 4:  # Зарядка
                if self.agent_position == self.base_position:
                    if self.energy > 30:
                        self.reward += const.PENALTY_BASE_BACK
                        logging.info("Агент летит на базу без вызова")
                    else:
                        self.energy = const.ENERGY_CAPACITY
                        self.reward += const.REWARD_BASE_BACK
                        self.last_progress_step = self.step_count
                        logging.info("Агент зарядился на базе")
                new_position = self.agent_position
            case 5:  # Полив
                self.water_tank -= const.WATER_CONSUMPTION
                self.energy -= const.ENERGY_CONSUMPTION_WATER
                if self.agent_position in self.target_positions:
                    logging.info(f"Полил цветок на позиции {self.agent_position}")
                    idx = self.target_positions.index(self.agent_position)
                    if self.watered_status[idx] == 0:
                        self.last_progress_step = self.step_count
                        self.watered_status[idx] = 1
                        if self.agent_position in known_unwatered_flowers:
                            self.reward += const.REWARD_WATER_KNOWN_FLOWER
                        else:
                            self.reward += const.REWARD_WATER_SUCCESS
                    else:
                        self.reward += const.PENALTY_WATER_FAIL_ALREADY_WATERED
                        logging.warning(
                            f"Агент попытался полить цветок, который уже полит на позиции {self.agent_position}")
                else:
                    self.reward += const.PENALTY_WATER_FAIL_NOT_ON_FLOWER
                    logging.warning(f"Агент попытался полить вне цветка на позиции {self.agent_position}")
                new_position = self.agent_position

            case _:
                new_position = self.agent_position

        # Проверяем, известна ли яма
        if new_position in self.hole_positions:
            if new_position in self.known_holes:
                self.reward += const.PENALTY_COLLISION * 2  # Увеличиваем штраф
                logging.warning(f"Агент попытался зайти в известную яму на позиции {new_position}")
                # Не обновляем позицию
                # new_position = self.agent_position
            else:
                self.reward += const.PENALTY_COLLISION
                logging.warning(f"Агент попал в неизвестную яму на позиции {new_position}")
                self.hole_fall_count += 1
                self.agent_position = new_position
                self.known_holes.add(new_position)
        else:
            self.agent_position = new_position

        # if not self.has_clear_target(new_position):
        #     self.reward += const.REWARD_UNNECESSARY_MOVE
        #     logging.warning(f"Агент не имеет четкой цели, штраф за ненужное движение на позицию {new_position}")

        # Обновление посещенных клеток
        if self.visited[self.agent_position] == 0:
            self.visited[self.agent_position] = 1
            self.last_progress_step = self.step_count

        if self.agent_position not in self.explored_cells:
            self.reward += const.REWARD_EXPLORE
            self.explored_cells.add(self.agent_position)

        # Дополнительное вознаграждение за нахождение рядом с неполитым цветком
        for pos in known_unwatered_flowers:
            distance = abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
            if distance == 1:
                self.reward += const.NEXT_2_UNWATERED_FLOWER
                break

        # Приоритизация неполитых цветов с увеличенным вознаграждением
        if np.any(self.watered_status == 0):
            if known_unwatered_flowers:
                nearest_flower = min(
                    known_unwatered_flowers,
                    key=lambda pos: abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
                )
                if nearest_flower not in self.known_flowers:
                    self.reward += 100  # Увеличенное вознаграждение

        # Проверка на завершение миссии
        terminated = False
        truncated = False
        info = {}

        if np.all(self.watered_status == 1):
            logging.info("Все цветы политы")
            self.agent_position = self.base_position
            logging.info("Агент вернулся на базу")
            self.reward += const.REWARD_COMPLETION
            terminated = True

        # if self.step_count - self.last_progress_step > MAX_STEPS_WITHOUT_PROGRESS:
        #     logging.info("Отсутствие прогресса в течение слишком большого количества шагов")
        #     terminated = True

        # if self.energy <= (0.2 * const.ENERGY_CAPACITY) and self.step_count - self.last_progress_step > 0:
        #     logging.info("Низкий уровень энергии без прогресса")
        #     self.reward += const.PENALTY_LOW_ENERGY_NO_PROGRESS

        # и так есть штраф за ямы
        # if self.hole_fall_count >= const.MAX_HOLE_FALL:
        #     logging.info("Агент слишком много раз попадал в ямы")
        #     self.reward += -50

        elapsed_time = time.perf_counter() - self.start_time
        # if elapsed_time > MAX_TIME and np.sum(self.watered_status) < MIN_FLOWERS_TO_WATER:
        #     logging.info("Время вышло, недостаточно полито цветов")
        #     reward += -100
        #     truncated = True

        # Проверка расстояния до ближайшего цветка
        if len(self.target_positions) > 0:
            unwatered_positions = [
                pos for idx, pos in enumerate(self.target_positions)
                if self.watered_status[idx] == 0
            ]
            if unwatered_positions:
                nearest_flower_dist = min([
                    abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
                    for pos in unwatered_positions
                ])
            else:
                nearest_flower_dist = 0
        else:
            nearest_flower_dist = 0

        if nearest_flower_dist > const.MAX_DISTANCE_FROM_FLORAL:
            self.steps_since_last_distance += 1
            if self.steps_since_last_distance >= const.MAX_STEPS_DISTANCE:
                logging.info("Агент слишком долго находится далеко от любого цветка")
                self.reward += const.REWARD_MAX_STEPS_DISTANCE
        else:
            self.steps_since_last_distance = 0

        if self.reward > 0:
            self.last_progress_step = self.step_count

        self.reward += const.REWARD_TIME(elapsed_time)
        self.reward += const.REWARD_STEPS(self.step_count)

        if self.step_count >= const.MAX_STEPS_GAME:
            logging.info("Достигнуто максимальное количество шагов")
            truncated = True

        obs = self._get_observation()
        logging.info(
            f"Шаг: {self.step_count},"
            f"Действие: {action}, "
            f"Награда: {self.reward}, "
            f"Завершено: {terminated}, "
            f"Прервано: {truncated}"
        )
        return obs, self.reward, terminated, truncated, info

    def has_clear_target(self, current_position):
        """
        Проверяет, есть ли у агента действительная цель в текущем положении.
        Если агент находится рядом с целью или есть цветок, который он пытается полить.
        """
        # Проверяем, находимся ли мы в пределах двух клеток от неполитого цветка.
        for target in self.target_positions:
            if self.watered_status[self.target_positions.index(target)] == 0:
                distance = abs(current_position[0] - target[0]) + abs(
                    current_position[1] - target[1])
                if distance <= 2:  # Можно настроить расстояние в зависимости от предпочтений
                    return True
        return False

    def render(self):
        self.screen.fill(const.GREEN)
        # Отрисовка сетки
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(self.screen, const.BLACK,
                                 (x * const.CELL_SIZE, y * const.CELL_SIZE, const.CELL_SIZE, const.CELL_SIZE),
                                 1)  # Рисуем черную границу вокруг каждой клетки

        # Отрисовка базы
        self.screen.blit(const.BASE_ICON,
                         (self.base_position[1] * const.CELL_SIZE, self.base_position[0] * const.CELL_SIZE))

        # Рисуем цветы и ямы, которые были обнаружены
        for i, pos in enumerate(self.target_positions):
            if pos in self.known_flowers:
                if self.watered_status[i]:
                    icon = const.WATERED_FLOWER_ICON
                else:
                    icon = const.FLOWER_ICON
                self.screen.blit(icon, (pos[1] * const.CELL_SIZE, pos[0] * const.CELL_SIZE))

        for hole in self.hole_positions:
            if hole in self.known_holes:
                self.screen.blit(const.HOLE_ICON, (hole[1] * const.CELL_SIZE, hole[0] * const.CELL_SIZE))

        # Рисуем линию к ближайшему цветку
        known_unwatered_flowers = [
            pos for idx, pos in enumerate(self.target_positions)
            if self.watered_status[idx] == 0 and pos in self.known_flowers
        ]
        if known_unwatered_flowers:
            nearest_flower = min(
                known_unwatered_flowers,
                key=lambda pos: abs(self.agent_position[0] - pos[0]) + abs(self.agent_position[1] - pos[1])
            )
            agent_x = self.agent_position[1] * const.CELL_SIZE + const.CELL_SIZE // 2
            agent_y = self.agent_position[0] * const.CELL_SIZE + const.CELL_SIZE // 2
            flower_x = nearest_flower[1] * const.CELL_SIZE + const.CELL_SIZE // 2
            flower_y = nearest_flower[0] * const.CELL_SIZE + const.CELL_SIZE // 2
            pygame.draw.line(self.screen, const.BLUE, (agent_x, agent_y), (flower_x, flower_y), 2)

        # Накладываем исследование области
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if pos not in self.explored_cells:
                    dark_overlay = pygame.Surface((const.CELL_SIZE, const.CELL_SIZE), pygame.SRCALPHA)
                    dark_overlay.fill((0, 0, 0, 200))  # Непрозрачный
                    self.screen.blit(dark_overlay, (y * const.CELL_SIZE, x * const.CELL_SIZE))

        # Отрисовка времени, очков, заряда и уровня воды
        elapsed_time = time.time() - self.start_time  # Рассчитываем время
        font = pygame.font.SysFont(None, const.FONT_SIZE)
        status_bar_height = 120  # Высота панели статуса
        status_bar_rect = pygame.Rect(0, const.SCREEN_SIZE, const.SCREEN_SIZE,
                                      status_bar_height)  # Прямоугольник для панели статуса
        pygame.draw.rect(self.screen, const.WHITE, status_bar_rect)

        self.screen.blit(font.render(f"Время: {elapsed_time:.2f} сек", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Очки: {int(self.reward)}", True, const.BLACK),
                         (10, const.SCREEN_SIZE + 40))
        self.screen.blit(font.render(f"Энергия: {self.energy}", True, const.BLACK),
                         (200, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Вода: {self.water_tank}", True, const.BLACK),
                         (200, const.SCREEN_SIZE + 40))
        self.screen.blit(font.render(f"Шаги: {self.step_count}", True, const.BLACK),
                         (400, const.SCREEN_SIZE + 10))
        self.screen.blit(font.render(f"Обнаружено ям: {len(self.known_holes)}/{const.COUNT_HOLES}",
                                     True, const.BLACK), (400, const.SCREEN_SIZE + 40))
        self.screen.blit(
            font.render(f"Обнаружено цветков: {len(self.known_flowers)}/{const.COUNT_FLOWERS}",
                        True, const.BLACK), (10, const.SCREEN_SIZE + 70))
        self.screen.blit(font.render(f"Полито цветков: {int(np.sum(self.watered_status))}/"
                                     f"{const.COUNT_FLOWERS}", True, const.BLACK), (300, const.SCREEN_SIZE + 70))

        # Отрисовка агента
        self.screen.blit(const.AGENT_ICON, (self.agent_position[1] * const.CELL_SIZE,
                                            self.agent_position[0] * const.CELL_SIZE))

        pygame.display.flip()
        pygame.time.wait(10)

    def render_message(self, render_text: str):
        """
        Display message in the center of screen
        :param render_text: str
        :return:
        """
        self.screen.fill(const.BLACK)
        text_surf = pygame.font.SysFont(None, const.TITLE_SIZE).render(render_text, True, const.GREEN)
        self.screen.blit(text_surf, text_surf.get_rect(center=(const.SCREEN_SIZE // 2, const.SCREEN_SIZE // 2)))
        pygame.display.flip()
