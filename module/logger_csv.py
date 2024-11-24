import csv
import os
from math import ceil

from config import csv_log


def log_to_csv(step_count: int, mission: int, total_reward: int, done_status: int, filename=csv_log):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')  # Используем ';' как разделитель
        if not file_exists:
            writer.writerow(['Step Count', 'Mission', 'Reward Sum', 'Done Status Sum'])
        # done_status_sum = np.sum(done_status) if isinstance(done_status, np.ndarray) else done_status
        writer.writerow([step_count, mission, ceil(total_reward), done_status, ''])
