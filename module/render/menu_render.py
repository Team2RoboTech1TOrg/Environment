import sys
from math import ceil
from typing import Any

import pygame
from pygame import Surface

import const


def render_text(screen: Surface, text: Any, font: pygame.font, color: tuple[int, int, int], x: int, y: int):
    """
    Render text on pygame screen.
    :param screen:
    :param text:
    :param font:
    :param color:
    :param x:
    :param y:
    :return:
    """
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


# TO DO убрать магические числа
def input_screen():
    """Pygame box for choosing number of agents, grid size and scenario"""
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Ввод параметров сценария")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    inputs = ["Введите количество агентов:",
              "Введите размер поля (минимум):",
              "Выберите сценарий:"
              " 1 - spraying"
              " 2 - exploration"]
    input_boxes = [pygame.Rect(150, 150 + i * 80, 300, 40) for i in range(len(inputs))]
    # scenarios = ["spraying", "exploration"]
    # dropdown = Dropdown(450, 390, 100, 50, scenarios)
    input_values = ["", "", ""]

    active_box = 0
    finished = False

    while not finished:
        screen.fill(const.GRAY)
        try:
            # Рассчитываем минимальный размер поля, если количество агентов введено
            num_agents = int(input_values[0]) if input_values[0].isdigit() else const.NUM_AGENTS
            grid_size_min = ceil(num_agents * 3 ** 0.5) + const.STATION_SIZE * 2
        except ValueError:
            grid_size_min = 0  # Если ввод некорректный

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, box in enumerate(input_boxes):
                    if box.collidepoint(event.pos):
                        active_box = i
            if event.type == pygame.KEYDOWN:
                if active_box < len(inputs):  # Убедиться, что активная коробка существует
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    elif event.key == pygame.K_RETURN:  # Перейти к следующему полю ввода
                        active_box += 1
                        if active_box >= len(inputs):
                            finished = True
                    else:
                        input_values[active_box] += event.unicode
            # dropdown.handle_event(event) #new

        # Отрисовка подсказок и полей ввода
        for i, text in enumerate(inputs):
            if i == 1 and grid_size_min > 0:
                text = f"Введите размер поля (от {grid_size_min}):"

            # Отображаем текст с выравниванием
            render_text(screen, text, small_font, (200, 200, 200), 150, 120 + i * 80)
            color = const.WHITE if i == active_box else const.BLACK
            pygame.draw.rect(screen, color, input_boxes[i], 2)
            render_text(screen, input_values[i], font, const.WHITE, input_boxes[i].x + 5, input_boxes[i].y + 5)
        # dropdown.draw(screen) # new
        pygame.display.flip()
        pygame.time.Clock().tick(30)
    pygame.quit()

    # Преобразование и проверка введённых данных
    try:
        num_agents = int(input_values[0]) if input_values[0] else const.NUM_AGENTS
        grid_size = int(input_values[1]) if input_values[1] else const.GRID_SIZE
        if grid_size < grid_size_min:
            raise ValueError(f"Размер поля должен быть больше, чем {grid_size_min}")
        selected_scenario = int(input_values[2]) if input_values[2] else 1
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        sys.exit()

    return num_agents, grid_size, selected_scenario
