import time

import pygame
import sys
from stable_baselines3 import PPO

import const as c
from environments.FarmingEnv import FarmingEnv
from config import log_dir
from logging_system.logger import logging
from logging_system.logger_csv import log_to_csv
from model_train import TrainingModel
from policy import CustomPolicy
from render.menu_render import input_screen
from scenarios.scenarios_dict import get_dict_scenarios


def run():
    num_agents, grid_size, selected = input_screen()
    scenarios = get_dict_scenarios(num_agents, grid_size)
    selected_scenario = scenarios.get(selected)

    if not selected_scenario:
        print(f"Ошибка: сценарий с номером {selected} не найден. Выбран сценарий по умолчанию.")
        selected_scenario = scenarios[1]
    try:
        env = FarmingEnv(selected_scenario)
        train = TrainingModel(env, render=True)
        train.train_model()
        train.save_model()
        time.sleep(2)
        model = train.get_model()

        clock = pygame.time.Clock()
        pygame.display.set_caption(selected_scenario.__str__())
        obs, info = env.reset()
        step_count = 1
        log_status = True
        mission = 1
        total_reward = 0
        while mission < 9:#True: # depends of log status
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)
            pygame.time.wait(10)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if log_status:
                log_to_csv(mission, step_count, int(reward), int(total_reward), info['done'], action, info['agent'])
            env.render()
            step_count += 1
            if truncated:
                obs, info = env.reset()
                message = f"Новая миссия {mission}"
                env.render_message(message)
                time.sleep(5)
                step_count = 1
                mission += 1
                total_reward = 0
            if terminated:
                message = f"Конец миссии\n\n награда: {int(total_reward)}\n шагов: {step_count}"
                env.render_message(message)
                time.sleep(5)
                if log_status:
                    obs, info = env.reset()
                    step_count = 1
                    mission += 1
                    total_reward = 0
                    message = f"Новая миссия {mission}"
                    env.render_message(message)
                    time.sleep(5)
                else:
                    break
            clock.tick(15)  # slow
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
