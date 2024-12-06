from math import ceil

import const
from environments.FarmingEnv import FarmingEnv
from logging_system.logger import logging
from model.TestingModel import TestingModel
from model.TrainingModel import TrainingModel
from scenarios.scenarios_dict import get_dict_scenarios


def run_server():
    print("Введите количество агентов:")
    num_agents = int(input()) or const.NUM_AGENTS
    print(f"Введите размер поля больше, чем :"
          f"{ceil((num_agents + const.STATION_SIZE * 2 + const.MARGIN_SIZE * 2) / (1 - (const.OBSTACLE_PERCENT + const.TARGET_PERCENT)))}")

    grid_size = int(input()) or const.GRID_SIZE
    print("Выберите сценарий:")
    selected = int(input()) or 1

    scenarios = get_dict_scenarios(num_agents, grid_size)
    selected_scenario = scenarios.get(selected)

    if not selected_scenario:
        print(f"Ошибка: сценарий с номером {selected} не найден. Выбран сценарий по умолчанию.")
        selected_scenario = scenarios[1]

    print("Для обучения модели нажите - 1\n Для просмотра работы модели - 2\n Тестирование - 3")
    selected_mode = int(input()) or 1
    try:
        env = FarmingEnv(selected_scenario)
        train = TrainingModel(env)
        if selected_mode == 1:
            train.train_model()
            train.save_model()
        elif selected_mode == 2:
            model = train.get_model()
            test = TestingModel(env, model, log=True)  # логи в файл csv, 8 missions default
            test.test_model()
        else:
            model = train.train_model()
            test = TestingModel(env, model, log=True)
            test.test_model()
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
