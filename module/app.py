import time

import pygame
import sys
from stable_baselines3 import PPO, DDPG

import const
from FarmingEnv import FarmingEnv
from scenarios.ExplorationScenario import ExplorationScenario
from scenarios.SprayingScenario import SprayingScenario
from config import log_dir
from logger import logging
from render.menu_render import input_screen


def run():
    num_agents, grid_size, selected = input_screen()
    # TO DO сделать например словарь сценариев отдельно
    spraying_scenario = SprayingScenario(num_agents, grid_size)
    exploration_scenario = ExplorationScenario(num_agents, grid_size)
    scenarios = {
        1: spraying_scenario,
        2: exploration_scenario
    }
    selected_scenario = scenarios.get(selected)
    if not selected_scenario:
        print(f"Ошибка: сценарий с номером {selected} не найден. Выбран сценарий по умолчанию.")
        selected_scenario = exploration_scenario
    try:
        env = FarmingEnv(selected_scenario)
        hyperparameters_message = (
            f"Гиперпараметры модели:\n\n"
            f"Темп: {const.LEARNING_RATE}\n"
            f"Гамма: {const.GAMMA}\n"
            f"Диапазон обрезки: {const.CLIP_RANGE}\n"
            f"Длина эпизода: {const.N_STEPS}\n"
            f"Энтропия: {const.COEF}\n"
            f"Баланс ценности: {const.VF_COEF}\n"
            f"Эпох: {const.N_EPOCHS}\n"
            f"Размер батча: {const.BATCH_SIZE}\n"
        )

        message = "Начало обучения модели\n\n\n" + hyperparameters_message
        env.render_message(message)
        pygame.display.set_caption("OS SWARM OF DRONES")
        logging.info(message)
        policy = 'MultiInputPolicy' #'MlpPolicy'
        model = PPO(
            policy,
            env,
            learning_rate=const.LEARNING_RATE,
            gamma=const.GAMMA,
            clip_range=const.CLIP_RANGE,
            n_steps=const.N_STEPS,
            ent_coef=const.COEF,
            verbose=1,
            vf_coef=const.VF_COEF,
            n_epochs=const.N_EPOCHS,
            batch_size=const.BATCH_SIZE,
            tensorboard_log=log_dir,
        )
        model.learn(total_timesteps=const.TIME)
        message = "Обучение модели\nзавершено."
        logging.info(message)
        env.render_message(message)
        model.save(f"{selected_scenario}_model")
        time.sleep(2)
        # model = PPO.load("spraying_scenario_model", print_system_info=True)
        clock = pygame.time.Clock()
        pygame.display.set_caption(selected_scenario.__str__())
        obs, info = env.reset()
        step_count = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)
            pygame.time.wait(10)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step_count += 1
            if truncated:
                obs, info = env.reset()
                message = f"Новая миссия"  # add counter games
                env.render_message(message)
                time.sleep(5)
                step_count = 0

            if terminated:
                message = f"Конец миссии\n\n награда: {int(reward)}\n шагов: {step_count}"
                env.render_message(message)
                time.sleep(5)
                break
            clock.tick(15)  # slow
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
