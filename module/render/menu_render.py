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
    """
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def input_screen() -> tuple[int, int, int, int]:
    """
    Pygame box for choosing number of agents, grid size, and scenario.
    :return: tuple: List of selected operation, number of agents, size of field and scenario number.
    """
    pygame.init()
    display = 600
    screen = pygame.display.set_mode((display, display))
    width = display // 2
    pygame.display.set_caption("Ввод параметров миссии")
    font = pygame.font.SysFont(c.FONT, 30)
    small_font = pygame.font.SysFont(c.FONT, 24)

    # Поля ввода
    inputs = ["Введите количество агентов:",
              "Введите размер поля (минимум):",
              "Выберите сценарий:"]
    center_x = display // 2 - width // 2  # Центр по горизонтали
    start_y = 250  # Начальная вертикальная позиция для первого поля
    input_boxes = [
        pygame.Rect(center_x, start_y, width, 40),  # Первое поле
        pygame.Rect(center_x, start_y + 80, width, 40),  # Второе поле
        pygame.Rect(center_x, start_y + 160, width, 40),  # Третье поле
    ]
    input_values = ["", "", "1"]  # Значения полей по умолчанию

    # Радио-кнопки
    radio_buttons = [
        # {"label": "Тестирование", "pos": (50, 150), "selected": True},  # TO DO только для наших тестов
        {"label": "Обучение модели", "pos": (50, 100), "selected": False},
        {"label": "Отрисовка результатов", "pos": (50, 50), "selected": False},
    ]

    # Выпадающий список
    scenarios = ["1 - spraying", "2 - exploration"]#, "3 - animal map"]
    dropdown_open = False
    dropdown_scroll_offset = 0
    dropdown_visible_count = 3

    active_box = 0
    finished = False
    grid_size_min = 0

    while not finished:
        screen.fill(c.GRAY)
        # Рассчитать минимальный размер поля на основе текущих значений
        num_agents = int(input_values[0]) if input_values[0].isdigit() else c.NUM_AGENTS
        grid_size_min = ceil((num_agents + c.STATION_SIZE * 2 + c.MARGIN_SIZE * 2) / (
                1 - (c.OBSTACLE_PERCENT + c.TARGET_PERCENT)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Обрабатываем только левый клик

                    # Обработка радио-кнопок
                    for button in radio_buttons:
                        button_rect = pygame.Rect(button["pos"][0], button["pos"][1], 20, 20)
                        if button_rect.collidepoint(event.pos):
                            for rb in radio_buttons:
                                rb["selected"] = False
                            button["selected"] = True

                    # Обработка клика по полям ввода
                    for i, box in enumerate(input_boxes):
                        if box.collidepoint(event.pos):
                            active_box = i

                    # Обработка клика по выпадающему списку
                    if dropdown_open:
                        dropdown_box = pygame.Rect(input_boxes[2].x, input_boxes[2].bottom, width,
                                                   40 * dropdown_visible_count)
                        if dropdown_box.collidepoint(event.pos):
                            clicked_index = (event.pos[1] - input_boxes[2].bottom) // 40 + dropdown_scroll_offset
                            if 0 <= clicked_index < len(scenarios):
                                input_values[2] = str(clicked_index + 1)
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
                if active_box < len(inputs) and not dropdown_open:
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

        # Отрисовка радио-кнопок
        for button in radio_buttons:
            button_rect = pygame.Rect(button["pos"][0], button["pos"][1], 20, 20)
            pygame.draw.rect(screen, c.WHITE, button_rect, 2)
            if button["selected"]:
                pygame.draw.circle(screen, c.WHITE, (button["pos"][0] + 10, button["pos"][1] + 10), 6)
            render_text(screen, button["label"], small_font, c.WHITE, button["pos"][0] + 30, button["pos"][1] - 5)

        # Отрисовка полей ввода
        for i, box in enumerate(input_boxes):
            color = c.WHITE if i == active_box else c.BLACK
            pygame.draw.rect(screen, color, box, 2)

            if i == 1 and grid_size_min > 0:
                text = f"Введите размер поля (от {grid_size_min}):"
            else:
                text = inputs[i]

            render_text(screen, text, small_font, c.WHITE, box.x, box.y - 30)
            render_text(screen, input_values[i], font, c.WHITE, box.x + 5, box.y + 5)

        # Отрисовка выпадающего списка
        if dropdown_open:
            dropdown_box = pygame.Rect(input_boxes[2].x, input_boxes[2].bottom, width, 40 * dropdown_visible_count)
            pygame.draw.rect(screen, c.LIGHT_GRAY, dropdown_box)
            pygame.draw.rect(screen, c.BLACK, dropdown_box, 2)

            visible_scenarios = scenarios[dropdown_scroll_offset:dropdown_scroll_offset + dropdown_visible_count]
            for i, scenario in enumerate(visible_scenarios):
                option_box = pygame.Rect(input_boxes[2].x, input_boxes[2].bottom + i * 40, width, 40)
                pygame.draw.rect(screen, c.LIGHT_GRAY, option_box)
                pygame.draw.rect(screen, c.BLACK, option_box, 1)
                render_text(screen, scenario, small_font, c.WHITE, option_box.x + 5, option_box.y + 5)

        pygame.display.flip()
        pygame.time.Clock().tick(30)
    pygame.quit()

    try:
        num_agents = int(input_values[0]) if input_values[0] else c.NUM_AGENTS
        grid_size = int(input_values[1]) if input_values[1] else c.GRID_SIZE
        if grid_size < grid_size_min:
            raise ValueError(f"Размер поля должен быть больше, чем {grid_size_min}")
        selected_scenario = int(input_values[2]) if input_values[2] else 1
        selected_mode = next((button["label"] for button in radio_buttons if button["selected"]), None)
        selected_mode = handle_selected_radio_button(selected_mode)
        if not selected_mode:
            raise ValueError("Не выбран режим работы.")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        sys.exit()
    return selected_mode, num_agents, grid_size, selected_scenario


def handle_selected_radio_button(selected_label) -> int:
    label = 1#3
    if selected_label == "Отрисовка результатов":
        label = 2
    elif selected_label == "Обучение модели":
        label = 1
    # elif selected_label == "Тестирование":
    #     label = 3
    return label
