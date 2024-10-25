import pygame
import sys
import numpy as np
from stable_baselines3 import PPO

from CONST import WATER_CONSUMPTION, WATER_CAPACITY, SCREEN_SIZE
from WateringEnv import WateringEnv
from OptimalSearchAgent import OptimalSearchAgent


def get_action(current_pos, next_step):
    if next_step[0] < current_pos[0]:
        return 0  # Движение вверх
    elif next_step[0] > current_pos[0]:
        return 1  # Движение вниз
    elif next_step[1] < current_pos[1]:
        return 2  # Движение влево
    elif next_step[1] > current_pos[1]:
        return 3  # Движение вправо
    return None  # Нет движения


def run():
    screen = pygame.display.set_mode((SCREEN_SIZE,
                                      SCREEN_SIZE + 100))  # Создание окна с размерами экрана и дополнительной областью для отображения информации
    clock = pygame.time.Clock()  # Инициализация объекта для отслеживания времени

    env = WateringEnv()
    # Эти две строчки запускают ППО. Если их закомитить, и раскомитить то, что ниже, будет простой алгоритм
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100)

    # agent = OptimalSearchAgent(env)  # Создание объекта агента, отвечающего за поиск оптимальных путей
    #
    # # Основной цикл симуляции
    # obs = env.reset()  # Сбрасываем среду, получаем начальное наблюдение
    # done = False  # Инициализация переменной завершения эпизода
    #
    # # Основной игровой цикл
    # while True:
    #     # Обрабатываем события Pygame
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #
    #     # Определяем цель
    #     if env.water_tank < WATER_CONSUMPTION:
    #         goal = (5, 5)  # Возвращаемся на базу для заправки водой
    #     else:
    #         remaining_targets = [pos for i, pos in enumerate(env.target_positions) if env.watered_status[i] == 0]
    #         if remaining_targets:
    #             goal = remaining_targets[0]  # Следующий неполитый цветок
    #         else:
    #             goal = (5, 5)  # Все цветы политы, возвращаемся на базу
    #
    #             # Проверяем, достиг ли агент базы после выполнения всех задач
    #             if np.array_equal(env.agent_position, goal):
    #                 # Отображаем сообщение и предлагаем выйти из игры
    #                 print("Все задачи выполнены! Нажмите 'Esc' для выхода или 'Enter' чтобы продолжить.")
    #                 waiting_for_exit = True
    #                 while waiting_for_exit:
    #                     for event in pygame.event.get():
    #                         if event.type == pygame.QUIT:
    #                             pygame.quit()
    #                             sys.exit()
    #                         elif event.type == pygame.KEYDOWN:
    #                             if event.key == pygame.K_ESCAPE:
    #                                 pygame.quit()
    #                                 sys.exit()
    #                             elif event.key == pygame.K_RETURN:
    #                                 pygame.quit()
    #                                 sys.exit()
    #
    #     # Ищем оптимальный путь к цели
    #     path = agent.find_optimal_path(goal)
    #     print(f"Текущая позиция: {env.agent_position}, Цель: {goal}, Путь: {path}")
    #
    #     # Определяем действие
    #     action = None
    #     if np.array_equal(env.agent_position, goal):
    #         if np.array_equal(goal, (5, 5)) and env.water_tank < WATER_CAPACITY:
    #             env.water_tank = WATER_CAPACITY  # Заправляем бак водой
    #             print("Заправка водой на базе.")
    #         elif any(np.array_equal(goal, pos) for pos in env.target_positions):
    #             action = 4  # Поливаем цветок
    #             print("Полив цветка.")
    #     else:
    #         if path:
    #             next_step = path[0]
    #             action = get_action(env.agent_position, next_step)
    #
    #     # Выполняем действие, если оно определено
    #     if action is not None:
    #         obs, reward, done, info = env.step(action)
    #         print(f"Действие: {action}, Новая позиция: {env.agent_position}, Награда: {reward}")
    #
    #     # Отрисовываем обновленное состояние среды
    #     env.render(screen)
    #
    #     # Ограничиваем скорость симуляции до 5 кадров в секунду
    #     clock.tick(5)