import csv
import os

from gymnasium.core import ActType

from config import csv_log


def log_to_csv(mission: int, step_count: int, reward: int, total_reward: int, done_status: int, action: ActType,
               agent: int, filename=csv_log):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(['mission', 'step', 'reward', 'cum_reward', 'done_sum', 'action', 'agent'])
        writer.writerow([mission, step_count, reward, total_reward, done_status, action, agent])
