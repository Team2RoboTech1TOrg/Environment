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


def convert_to_multidiscrete(action_spaces: spaces.Dict) -> spaces.MultiDiscrete:
    """
    Переводит формат словаря в нужный для мультиагентной среды формат MultiDiscrete
    :param action_spaces:
    :return:
    """
    discrete_spaces = list(action_spaces.values())
    nvec = [space.n for space in discrete_spaces]
    new_action_space = spaces.MultiDiscrete(nvec)
    return new_action_space



