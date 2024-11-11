import pygame
from gymnasium import spaces

from logger import logging


def load_image(filename: str, cell_size: int):
    """
    Функция загрузки и масштабирования изображения для объектов
    """
    try:
        image = pygame.image.load(filename)
        return pygame.transform.scale(image, (cell_size, cell_size))
    except pygame.error as e:
        logging.error(f"Не удалось загрузить изображение {filename}: {e}")
        return pygame.Surface((cell_size, cell_size))  # Возвращаем пустую поверхность


def convert_to_multidiscrete(action_space):
    discrete_spaces = list(action_space.values())
    nvec = [space.n for space in discrete_spaces]
    new_action_space = spaces.MultiDiscrete(nvec)
    return new_action_space



