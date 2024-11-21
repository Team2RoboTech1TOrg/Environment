import sys
from math import ceil
from typing import Any

import pygame
from pygame import Surface

import const as c


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
    display = 600
    screen = pygame.display.set_mode((display, display))
    rect_size = display // 4
    width = display // 2
    pygame.display.set_caption("Ввод параметров сценария")
    font = pygame.font.SysFont(c.FONT, 30)
    small_font = pygame.font.SysFont(c.FONT, 24)

    inputs = ["Введите количество агентов:",
              "Введите размер поля (минимум):",
              "Выберите сценарий:"]
    input_boxes = [pygame.Rect(rect_size, rect_size + i * 80, width, 40) for i in range(len(inputs))]
    input_values = ["", "", "1"]  # По умолчанию выбран сценарий 1

    # Данные для выпадающего списка
    scenarios = ["1 - spraying", "2 - exploration"]
    dropdown_open = False
    dropdown_scroll_offset = 0  # Смещение прокрутки списка
    dropdown_visible_count = 3  # Количество видимых опций в списке

    active_box = 0
    finished = False
    grid_size_min = 0

    while not finished:
        screen.fill(c.GRAY)
        # расчет исходя из доли объектов, кол-ва агентов, отступа и ширины базы
        num_agents = int(input_values[0]) if input_values[0].isdigit() else c.NUM_AGENTS
        grid_size_min = ceil((num_agents + c.STATION_SIZE * 2 + c.MARGIN_SIZE * 2) / (
                    c.OBSTACLE_PERCENT + c.TARGET_PERCENT))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Обрабатываем только левый клик мыши
                    if dropdown_open:
                        # Если список открыт, проверяем, кликнул ли пользователь в него
                        dropdown_box = pygame.Rect(rect_size, input_boxes[2].bottom, width, 40 * dropdown_visible_count)
                        if dropdown_box.collidepoint(event.pos):
                            # Определяем индекс выбранного элемента
                            clicked_index = (event.pos[1] - input_boxes[2].bottom) // 40 + dropdown_scroll_offset
                            if 0 <= clicked_index < len(scenarios):
                                input_values[2] = str(clicked_index + 1)  # Сохраняем выбор
                        dropdown_open = False  # Закрываем список, если клик не внутри

                    else:
                        # Проверяем клик в другие поля ввода
                        for i, box in enumerate(input_boxes):
                            if box.collidepoint(event.pos):
                                active_box = i
                                if i == 2:  # Открыть список при клике на поле сценария
                                    dropdown_open = True
                                    dropdown_scroll_offset = 0  # Сбрасываем прокрутку
                                else:
                                    dropdown_open = False  # Закрываем список при клике вне него

            if event.type == pygame.KEYDOWN:
                if active_box < len(inputs) and not dropdown_open:  # Ввод текста только в закрытом состоянии списка
                    if event.key == pygame.K_BACKSPACE:
                        input_values[active_box] = input_values[active_box][:-1]
                    elif event.key == pygame.K_RETURN:  # Переход к следующему полю
                        active_box += 1
                        if active_box >= len(inputs):
                            finished = True
                    else:
                        input_values[active_box] += event.unicode

            if event.type == pygame.MOUSEWHEEL:
                if dropdown_open:
                    # Прокрутка выпадающего списка
                    dropdown_scroll_offset -= event.y
                    dropdown_scroll_offset = max(0,
                                                 min(dropdown_scroll_offset, len(scenarios) - dropdown_visible_count))
                # Не добавляем действия при прокрутке колеса мыши, если выпадающий список не открыт
                else:
                    pass  # Игнорируем другие действия для прокрутки

        # Отрисовка подсказок и полей ввода
        for i, text in enumerate(inputs):
            if i == 1 and grid_size_min > 0:
                text = f"Введите размер поля (от {grid_size_min}):"

            # Отображаем текст с выравниванием
            render_text(screen, text, small_font, c.WHITE, rect_size, 120 + i * 80)
            color = c.WHITE if i == active_box else c.BLACK
            pygame.draw.rect(screen, color, input_boxes[i], 2)
            render_text(screen, input_values[i], font, c.WHITE, input_boxes[i].x + 5, input_boxes[i].y + 5)

        # Отрисовка выпадающего списка
        if dropdown_open:
            dropdown_box = pygame.Rect(rect_size, input_boxes[2].bottom, width, 40 * dropdown_visible_count)
            pygame.draw.rect(screen, c.LIGHT_GRAY, dropdown_box)
            pygame.draw.rect(screen, c.BLACK, dropdown_box, 2)  # Граница списка

            # Отображаем видимые элементы
            visible_scenarios = scenarios[dropdown_scroll_offset:dropdown_scroll_offset + dropdown_visible_count]
            for i, scenario in enumerate(visible_scenarios):
                option_box = pygame.Rect(rect_size, input_boxes[2].bottom + i * 40, width, 40)
                pygame.draw.rect(screen, c.LIGHT_GRAY, option_box)
                pygame.draw.rect(screen, c.BLACK, option_box, 1)
                render_text(screen, scenario, small_font, c.WHITE, option_box.x + 5, option_box.y + 5)

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
