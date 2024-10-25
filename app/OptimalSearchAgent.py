from queue import PriorityQueue

from CONST import GRID_SIZE

# Определение класса OptimalSearchAgent для поиска оптимальных путей
class OptimalSearchAgent:
    def __init__(self, env):
        self.env = env  # Инициализация среды

    # Эвристическая функция для оценки расстояния до цели (Манхэттенское расстояние)
    def heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])  # Возвращаем сумму абсолютных разностей по координатам

    # Реализация алгоритма поиска A*
    def a_star_search(self, start, goal):
        open_set = PriorityQueue()  # Инициализация приоритетной очереди для хранения возможных путей
        open_set.put((0, start))  # Добавляем стартовую позицию с приоритетом 0
        came_from = {}  # Словарь для хранения пути к каждой позиции
        g_score = {start: 0}  # Стоимость пути до начальной точки (0)

        while not open_set.empty():  # Пока есть нерассмотренные пути
            _, current = open_set.get()  # Извлекаем элемент с наименьшей стоимостью

            if current == goal:  # Если достигли цели
                path = []  # Инициализация списка пути
                while current in came_from:  # Восстанавливаем путь
                    path.append(current)  # Добавляем текущую точку в путь
                    current = came_from[current]  # Переходим к предыдущей точке
                path.reverse()  # Переворачиваем путь для получения правильной последовательности
                return path  # Возвращаем найденный путь

            # Проход по соседям текущей позиции
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Возможные направления (вверх, вниз, влево, вправо)
                neighbor = (current[0] + dx, current[1] + dy)  # Рассчитываем координаты соседа
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:  # Проверяем, находится ли сосед в пределах сетки
                    tentative_g_score = g_score[current] + 1  # Стоимость перехода к соседу
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:  # Если новый путь короче
                        came_from[neighbor] = current  # Обновляем путь
                        g_score[neighbor] = tentative_g_score  # Обновляем стоимость
                        f_score = tentative_g_score + self.heuristic(neighbor, goal)  # Рассчитываем приоритет с учетом эвристики
                        open_set.put((f_score, neighbor))  # Добавляем соседа в очередь с приоритетом

        return []  # Если путь не найден, возвращаем пустой список

    # Метод поиска оптимального пути к цели
    def find_optimal_path(self, goal):
        return self.a_star_search(self.env.agent_position, tuple(goal))  # Возвращаем путь от текущей позиции до цели