import time

import pygame
import sys
from stable_baselines3 import PPO

from const import LEARNING_RATE, GAMMA, CLIP_RANGE, N_STEPS, COEF, MAX_STEPS_GAME, N_EPOCHS, BATCH_SIZE, CLIP_RANGE_VF
from WateringEnv import WateringEnv
from config import log_dir
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
        model.save("ppo_watering_model")
        time.sleep(2)
        # model = PPO.load("ppo_watering_model", print_system_info=True)
        clock = pygame.time.Clock()
        pygame.display.set_caption("Drone Watering Flowers")

        obs, info = env.reset()
        step_count = 0
        for _ in range(MAX_STEPS_GAME * 5):  # костыль
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = model.predict(obs)
            pygame.time.wait(5)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step_count += 1
            if truncated:
                obs, info = env.reset()
                message = f"Новая игра"  # add counter games
                env.render_message(message)
                time.sleep(5)
                step_count = 0

            if terminated:
                message = f"Конец игры, награда: {int(reward)}"
                env.render_message(message)
                time.sleep(5)
                break
            clock.tick(60)
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
