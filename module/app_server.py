import time
from math import ceil

# import pygame
from stable_baselines3 import PPO

import const
from environments.FarmingEnv import FarmingEnv
from config import log_dir
from logging_system.logger import logging
from logging_system.logger_csv import log_to_csv
from model_train import TrainingModel
from policy import CustomPolicy
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

    # print("Для обучения модели нажите - 1")
    try:
        env = FarmingEnv(selected_scenario)
        train = TrainingModel(env)
        train.train_model()
        train.save_model()
        time.sleep(2)
        model = train.get_model()

        obs, info = env.reset()
        step_count = 0
        log_status = True
        total_reward = 0
        mission = 1
        while mission < 9: #depends of log status
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if log_status:
                log_to_csv(mission, step_count, int(reward), int(total_reward), info['done'], action, info['agent'])
            env.render()
            step_count += 1
            if truncated:
                obs, info = env.reset()
                message = f"Новая миссия"
                logging.info(message)
                step_count = 0
                mission += 1
                total_reward = 0
            if terminated:
                message = f"Конец миссии\n\n награда: {int(total_reward)}\n шагов: {step_count}"
                logging.info(message)
                if log_status:
                    obs, info = env.reset()
                    step_count = 0
                    mission += 1
                    total_reward = 0
                else:
                    break
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise

