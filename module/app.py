import time
from math import ceil

import pygame
import sys
from stable_baselines3 import PPO

import const
from const import LEARNING_RATE, GAMMA, CLIP_RANGE, N_STEPS, COEF, MAX_STEPS_GAME, N_EPOCHS, BATCH_SIZE, CLIP_RANGE_VF
from WateringEnv import WateringEnv
from config import log_dir
from logger import logging


def run():
    print("Введите количество агентов:")
    num_agents = input() or const.NUM_AGENTS
    print(f"Введите размер поля больше, чем :"
          f"{ceil((const.COUNT_FLOWERS + const.COUNT_OBSTACLES + int(num_agents)) ** 0.5) + const.COUNT_STATION}")
    grid_size = input() or const.GRID_SIZE
    try:
        env = WateringEnv(int(num_agents), int(grid_size))
        message = "Начало обучения модели."
        env.render_message(message)
        pygame.display.set_caption("OS SWARM OF DRONES")
        logging.info(message)
        policy = 'MultiInputPolicy' #'MlpPolicy'
        model = PPO(
            policy,
            env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            clip_range=CLIP_RANGE,
            n_steps=N_STEPS,
            ent_coef=COEF,
            verbose=1,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            tensorboard_log=log_dir,
            clip_range_vf=CLIP_RANGE_VF
        )
        model.learn(total_timesteps=10000)
        message = "Обучение модели завершено."
        logging.info(message)
        env.render_message(message)
        model.save("spraying_scenario_model")
        time.sleep(2)
        # model = PPO.load("ppo_watering_model", print_system_info=True)
        clock = pygame.time.Clock()
        pygame.display.set_caption("Pesticide Spraying Scenario")

        obs, info = env.reset()
        step_count = 0
        for _ in range(MAX_STEPS_GAME * 5):  # костыль
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)
            pygame.time.wait(15)
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
                message = f"Конец миссии, награда: {int(reward)}, шагов: {step_count}"
                env.render_message(message)
                # time.sleep(5)
                break
            clock.tick(10)  # slow
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
