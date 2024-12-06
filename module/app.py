import time
import pygame

from environments.FarmingEnv import FarmingEnv
from logging_system.logger import logging
from model.TestingModel import TestingModel
from model.TrainingModel import TrainingModel
from render.menu_render import input_screen
from scenarios.scenarios_dict import get_dict_scenarios


def run():
    selected_mode, num_agents, grid_size, selected = input_screen()
    scenarios = get_dict_scenarios(num_agents, grid_size)
    selected_scenario = scenarios.get(selected)

    if not selected_scenario:
        print(f"Ошибка: сценарий с номером {selected} не найден. Выбран сценарий по умолчанию.")
        selected_scenario = scenarios[1]
    try:
        pygame.init()
        env = FarmingEnv(selected_scenario)
        train = TrainingModel(env, render_mode=True)
        if selected_mode == 1:
            train.train_model()
            train.save_model()
            time.sleep(2)
        elif selected_mode == 2:
            model = train.get_model()
            test = TestingModel(env, model, log=True, render_mode=True)
            test.test_model_render()
        else:
            model = train.train_model()
            test = TestingModel(env, model, log=True, render_mode=True)
            test.test_model_render()
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
