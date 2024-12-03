import csv
import os

from math import ceil
from gymnasium.core import ActType

from config import csv_log


def log_to_csv(mission: int, step_count: int, total_reward: int, done_status: int, action: ActType, filename=csv_log):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(['mission', 'step', 'reward', 'done_sum', 'action'])
        writer.writerow([mission, step_count, ceil(total_reward), done_status, action])
