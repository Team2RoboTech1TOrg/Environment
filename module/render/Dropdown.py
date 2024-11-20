import pygame
from pygame.locals import MOUSEBUTTONDOWN, KEYDOWN, K_UP, K_DOWN

import const


class Dropdown:
    def __init__(self, x, y, w, h, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected = 0
        self.droped = False
        self.font = pygame.font.SysFont(None, 30)

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.droped = not self.droped
            else:
                self.droped = False

        if event.type == KEYDOWN:
            if self.droped:
                if event.key == K_UP:
                    self.selected -= 1
                    if self.selected < 0:
                        self.selected = len(self.options) - 1
                elif event.key == K_DOWN:
                    self.selected += 1
                    if self.selected >= len(self.options):
                        self.selected = 0

    def draw(self, screen):
        pygame.draw.rect(screen, const.WHITE, self.rect)
        text = self.font.render(self.options[self.selected], True, const.BLACK)
        screen.blit(text, (self.rect.x + 10, self.rect.y + 5))

        if self.droped:
            for i in range(len(self.options)):
                rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width,
                                   self.rect.height)
                pygame.draw.rect(screen, const.GRAY, rect)
                text = self.font.render(self.options[i], True, const.BLACK)
                screen.blit(text, (rect.x + 10, rect.y + 5))
