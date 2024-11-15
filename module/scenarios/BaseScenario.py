from abc import ABC, abstractmethod


class BaseScenario(ABC):
    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode):
        pass

