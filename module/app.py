import time

import pygame
import sys
import numpy as np
from stable_baselines3 import PPO

from CONST import WATER_CONSUMPTION, LEARNING_RATE, GAMMA, CLIP_RANGE, N_STEPS, COEF, SCREEN_SIZE, RED, FONT_SIZE, BLACK
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
        model.learn(total_timesteps=1000)
        message = "Обучение модели завершено."
        logging.info(message)
        env.render_message(message)
        model.save("ppo_watering_model")
        time.sleep(2)

        clock = pygame.time.Clock()
        pygame.display.set_caption("Drone Watering Flowers")

        obs, info = env.reset()
        step_count = 0
        for _ in range(2000):  # while True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # Высокоуровневый контроллер
            if env.agent_position in env.target_positions:
                idx = env.target_positions.index(env.agent_position)
                if env.watered_status[idx] == 0 and env.water_tank >= WATER_CONSUMPTION:
                    action = 5  # Действие "Полив"
                    logging.debug(f"Иерархический контроллер: Полив цветка на позиции {env.agent_position}")
                else:
                    action, _ = model.predict(obs)
            else:
                action, _ = model.predict(obs)
                if action == 5:
                    # Заменяем действие "Полив" на другое допустимое действие, например, "Вверх" (0)
                    action = np.random.choice([0, 1, 2, 3, 4])
                    logging.debug("Иерархический контроллер: Действие 'Полив' недопустимо, заменено на другое действие")

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
            if terminated or truncated:
                obs, info = env.reset()
                message = f"Новая игра" # add counter games
                env.render_message(message)
                time.sleep(5)
                step_count = 0
            clock.tick(60)
        message = "Конец."
        env.render_message(message)
        time.sleep(5)
        env.close()
    except KeyboardInterrupt:
        logging.info("Прервано пользователем")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        raise
    finally:
        pygame.quit()
