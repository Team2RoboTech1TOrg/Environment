import sys
from math import ceil
from typing import Any

import pygame
from pygame import Surface

import const as c
from render.Dropdown import DropdownList


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
    """Pygame box for choosing number of agents, grid size, and scenario"""
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    rect_size = 150
    pygame.display.set_caption("Ввод параметров сценария")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    inputs = ["Введите количество агентов:",
              "Введите размер поля (минимум):",
              "Выберите сценарий:"]
    input_boxes = [pygame.Rect(rect_size, rect_size + i * 80, 300, 40) for i in range(len(inputs))]
    input_values = ["", "", "1"]  # По умолчанию выбран сценарий 1

    scenarios = ["1 - spraying", "2 - exploration"]
    dropdown_list = DropdownList(rect_size + 310, rect_size + 160, 200, 30, scenarios)

    active_box = 0
    finished = False

    while not finished:
        screen.fill(c.GRAY)
        # расчет исходя из доли объектов, кол-ва агентов, отступа и ширины базы
        num_agents = int(input_values[0]) if input_values[0].isdigit() else c.NUM_AGENTS
        grid_size_min = ceil((num_agents + c.STATION_SIZE * 2 + c.MARGIN_SIZE * 2) / (
                    c.OBSTACLE_PERCENT + c.TARGET_PERCENT))

        # Отображение всех элементов интерфейса
        for i, box in enumerate(input_boxes):
            pygame.draw.rect(screen, pygame.Color('white'), box)
            text = font.render(inputs[i], True, pygame.Color('black'))
            screen.blit(text, (box.x - 320, box.y + 12))
            if i != 2:  # Для обычных полей ввода
                value_text = small_font.render(input_values[i], True, pygame.Color('black'))
                screen.blit(value_text, (box.x + 10, box.y + 8))

        # Рисование и обработка событий для выпадающего списка
        dropdown_list.draw(screen, font)
        dropdown_open = dropdown_list.open

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if dropdown_open:
                        dropdown_list.handle_event(event)
                    else:
                        for i, box in enumerate(input_boxes[:2]):
                            if box.collidepoint(event.pos):
                                active_box = i
                                break
                        if input_boxes[2].collidepoint(event.pos):  # Клик на поле выбора сценария
                            dropdown_list.open = True
                            active_box = None
            if event.type == pygame.KEYDOWN:
                if active_box is not None:  # Ввод текста только в закрытом состоянии списка
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    elif event.key == pygame.K_RETURN:  # Переход к следующему полю
                        active_box += 1
                        if active_box >= len(inputs):
                            finished = True
                    else:
                        input_values[active_box] += event.unicode

        # Обновляем значение для выбранного сценария
        input_values[2] = str(dropdown_list.selected_option + 1)

        pygame.display.flip()
        pygame.time.Clock().tick(30)
    pygame.quit()

    # Преобразование и проверка введённых данных
    try:
        num_agents = int(input_values[0]) if input_values[0] else c.NUM_AGENTS
        grid_size = int(input_values[1]) if input_values[1] else c.GRID_SIZE
        if grid_size < grid_size_min:
            raise ValueError(f"Размер поля должен быть больше, чем {grid_size_min}")
        selected_scenario = int(input_values[2]) if input_values[2] else 1
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        sys.exit()

    return num_agents, grid_size, selected_scenario
