import pygame

from app import run
from app_server import run_server
server = True


if __name__ == '__main__':
    if server:
        pygame.init()
        run_server()
    else:
        run()

