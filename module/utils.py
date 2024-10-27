import pygame

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
