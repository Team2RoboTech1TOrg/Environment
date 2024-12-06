import sys
import time
import pygame

import const as c
from logging_system.logger_csv import log_to_csv
from logging_system.logger import logging


class TestingModel:
    def __init__(self, env, model, log=False, render_mode=False):
        self.env = env
        self.log_status = log
        self.model = model
        self.mission = 1
        self.step = 0
        self.total_reward = 0
        self.render_mode = render_mode

    def test_model(self):
        missions = 1
        if self.log_status:
            missions = c.MISSIONS_FOR_TEST
        obs, info = self.env.reset()
        while self.mission <= missions:
            action, _ = self.model.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            if self.log_status:
                log_to_csv(self.mission, self.step, int(reward), int(self.total_reward),
                           info['done'], action, info['agent'])
            self.step += 1
            if truncated or terminated:
                obs, info = self.terminate_mission(truncated)

    def test_model_render(self):
        clock = pygame.time.Clock()
        pygame.display.set_caption(self.env.scenario.__str__())
        obs, info = self.env.reset()
        missions = 1
        if self.log_status:
            missions = c.MISSIONS_FOR_TEST
        while self.mission <= missions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            action, _ = self.model.predict(obs)
            pygame.time.wait(10)
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(reward, self.total_reward)
            self.total_reward += reward
            if self.log_status:
                log_to_csv(self.mission, self.step, int(reward), int(self.total_reward),
                           info['done'], action, info['agent'])
            self.env.render()
            self.step += 1
            if truncated or terminated:
                obs, info = self.terminate_mission(truncated)
            clock.tick(15)

    def terminate_mission(self, truncated):
        if truncated:
            message = f"Новая миссия"
        else:
            message = f"Конец миссии\n\n награда: {int(self.total_reward)}\n шагов: {self.step}"
        obs, info = self.env.reset()
        logging.info(message)
        self.step, self.total_reward = 0, 0
        self.mission += 1
        if self.render_mode:
            self.env.render_message(message)
            time.sleep(5)
        return obs, info

