import pygame

from stable_baselines3 import PPO

import const as c
from config import log_dir
from logging_system.logger import logging
from policy import CustomPolicy


class TrainingModel:
    def __init__(self, env, render=False):
        self.env = env
        self.model = None
        self.log_dir = log_dir
        self.learning_rate = c.LEARNING_RATE
        self.gamma = c.GAMMA
        self.clip_range = c.CLIP_RANGE
        self.steps = c.N_STEPS
        self.entropy = c.COEF
        self.vf = c.VF_COEF
        self.epochs = c.N_EPOCHS
        self.batch = c.BATCH_SIZE
        self.total_steps = c.TIME
        self.gae_lambda = c.GAE
        self.render_mode = render

    def render_hyperparameters_message(self) -> str:
        if not self.render_mode:
            return ''
        message = (
            f"Гиперпараметры модели:\n\n"
            f"Темп: {self.learning_rate}\n"
            f"Гамма: {self.gamma}\n"
            f"Диапазон обрезки: {self.clip_range}\n"
            f"Длина эпизода: {self.steps}\n"
            f"Энтропия: {self.entropy}\n"
            f"Баланс ценности: {self.vf}\n"
            f"Эпох: {self.epochs}\n"
            f"Размер батча: {self.batch}\n"
        )
        return message

    def train_model(self):
        message = "Начало обучения модели\n\n\n" + self.render_hyperparameters_message()
        logging.info(message)

        if self.render_mode:
            self.env.render_message(message)
            pygame.display.set_caption("OS SWARM OF DRONES")

        self.model = PPO(
            CustomPolicy,
            self.env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            n_steps=self.steps,
            ent_coef=self.entropy,
            verbose=1,
            vf_coef=self.vf,
            n_epochs=self.epochs,
            batch_size=self.batch,
            tensorboard_log=self.log_dir,
            normalize_advantage=True,
        )
        self.model.learn(total_timesteps=self.total_steps)
        message = "Обучение модели\nзавершено."
        logging.info(message)
        if self.render_mode:
            self.env.render_message(message)
        pygame.quit()
        return self.model

    def save_model(self):
        self.model.save(f"{self.env.scenario}_model")
        message = "Модель сохранена."
        logging.info(message)

    def get_model(self):
        return PPO.load(f"{self.env.scenario}_model", print_system_info=True)
