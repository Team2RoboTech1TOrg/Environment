import os
import random
import pygame

from logging_system.logger import logging


def load_image(filename: str, cell_size: int):
    """
    Image loading and scaling function for objects.
    :param filename: Path to file
    :param cell_size: size of cell
    :return: Scaling image
    """
    try:
        image = pygame.image.load(filename)
        return pygame.transform.scale(image, (cell_size, cell_size))
    except pygame.error as e:
        logging.error(f"Не удалось загрузить изображение {filename}: {e}")
        return pygame.Surface((cell_size, cell_size))


def load_obstacles(directory, cell_size, count):
    """
    Load image files with obstacles.
    :param directory:
    :param cell_size:
    :param count:
    :return: List with images
    """
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = random.sample(all_files, min(count, len(all_files)))
    return [load_image(f, cell_size) for f in selected_files]


