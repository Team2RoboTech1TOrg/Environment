import time

import pygame
import sys
from stable_baselines3 import PPO

import const as c
from environments.FarmingEnv import FarmingEnv
from config import log_dir
from logging_system.logger import logging
from logging_system.logger_csv import log_to_csv
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
        hyperparameters_message = (
            f"Гиперпараметры модели:\n\n"
            f"Темп: {c.LEARNING_RATE}\n"
            f"Гамма: {c.GAMMA}\n"
            f"Диапазон обрезки: {c.CLIP_RANGE}\n"
            f"Длина эпизода: {c.N_STEPS}\n"
            f"Энтропия: {c.COEF}\n"
            f"Баланс ценности: {c.VF_COEF}\n"
            f"Эпох: {c.N_EPOCHS}\n"
            f"Размер батча: {c.BATCH_SIZE}\n"
        )

        message = "Начало обучения модели\n\n\n" + hyperparameters_message
        env.render_message(message)
        pygame.display.set_caption("OS SWARM OF DRONES")
        logging.info(message)
        policy = CustomPolicy
        model = PPO(
            policy,
            env,
            learning_rate=c.LEARNING_RATE,
            gamma=c.GAMMA,
            clip_range=c.CLIP_RANGE,
            n_steps=c.N_STEPS,
            ent_coef=c.COEF,
            verbose=1,
            vf_coef=c.VF_COEF,
            n_epochs=c.N_EPOCHS,
            batch_size=c.BATCH_SIZE,
            tensorboard_log=log_dir,
            normalize_advantage=True,
            policy_kwargs=c.policy_kwargs
        )
        model.learn(total_timesteps=c.TIME)
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
        log_status = True
        mission = 1
        while mission < 9:#True: # depends of log status
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)
            pygame.time.wait(10)
            obs, reward, terminated, truncated, info = env.step(action)
            if log_status:
                log_to_csv(mission, step_count, int(reward), info['done'], action)
            env.render()
            step_count += 1
            if truncated:
                obs, info = env.reset()
                message = f"Новая миссия {mission}"
                env.render_message(message)
                time.sleep(5)
                step_count = 0
                mission += 1
            if terminated:
                message = f"Конец миссии\n\n награда: {int(reward)}\n шагов: {step_count}"
                env.render_message(message)
                time.sleep(5)
                if log_status:
                    obs, info = env.reset()
                    step_count = 0
                    mission += 1
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
