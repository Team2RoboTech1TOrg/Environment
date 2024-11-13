class PointStatus:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'state: {self.name}>'


class Plant(PointStatus):
    def __init__(self):
        super().__init__('plant')


class Obstacle(PointStatus):
    def __init__(self):
        super().__init__('obstacle')


class Viewed:
    def __init__(self):
        super().__init__('viewed')


class Explored:
    def __init__(self):
        super().__init__('explored')
