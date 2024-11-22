import enum


class PointStatus(enum.Enum):
    empty = 0
    viewed = 1
    visited = 2


class ObjectStatus(enum.Enum):
    empty = 0
    obstacle = 1
    plant = 2
    base = 3

