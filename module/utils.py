import pygame


def load_image(filename: str, cell_size: int):
    """
    Функция загрузки и масштабирования изображения для объектов
    """
    image = pygame.image.load(filename)
    return pygame.transform.scale(image, (cell_size, cell_size))
