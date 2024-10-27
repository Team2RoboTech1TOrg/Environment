import time

import pygame
import sys
import numpy as np
from stable_baselines3 import PPO

from CONST import WATER_CONSUMPTION, LEARNING_RATE, GAMMA, CLIP_RANGE, N_STEPS, COEF, SCREEN_SIZE, RED, FONT_SIZE, \
    BLACK, MAX_STEPS_GAME
from WateringEnv import WateringEnv
from logger import logging


def run():
    try:
        env = WateringEnv()
        message = "Начало обучения модели."
        env.render_message(message)
        pygame.display.set_caption("Drone learning")
        logging.info(message)
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            clip_range=CLIP_RANGE,
            n_steps=N_STEPS,
            ent_coef=COEF,
            verbose=1
        )
        model.learn(total_timesteps=10000)
        message = "Обучение модели завершено."
        logging.info(message)
        env.render_message(message)
        model.save("ppo_watering_model")
        time.sleep(2)

        clock = pygame.time.Clock()
        pygame.display.set_caption("Drone Watering Flowers")

        obs, info = env.reset()
        step_count = 0
        for _ in range(MAX_STEPS_GAME * 5): #  костыль
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step_count += 1
            logging.info(
                f"Шаг: {step_count},"
                f"Действие: {action}, "
                f"Награда: {reward}, "
                f"Завершено: {terminated}, "
                f"Прервано: {truncated}"
            )
            if truncated:
                obs, info = env.reset()
                message = f"Новая игра" # add counter games
                env.render_message(message)
                time.sleep(5)
                step_count = 0

            if terminated:
                message = f"Конец игры, награда: {round(reward)}"
                env.render_message(message)
                time.sleep(5)
                env.close()
                break
            clock.tick(60)
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
