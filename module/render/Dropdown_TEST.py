import pygame


class DropdownList:
    def __init__(self, x, y, width, height, options, visible_count=3):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.visible_count = visible_count
        self.open = False
        self.scroll_offset = 0
        self.selected_option = 0

    def draw(self, screen, font):
        pygame.draw.rect(screen, pygame.Color('white'), self.rect)
        text = font.render(self.options[self.selected_option], True, pygame.Color('black'))
        screen.blit(text, (self.rect.x + 10, self.rect.y + 10))

        if self.open:
            dropdown_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                        self.rect.width, self.rect.height * self.visible_count)
            pygame.draw.rect(screen, pygame.Color('lightgray'), dropdown_rect)
            for i in range(min(self.visible_count, len(self.options))):
                option_text = font.render(self.options[i + self.scroll_offset], True, pygame.Color('black'))
                screen.blit(option_text, (dropdown_rect.x + 10, dropdown_rect.y + 10 + i * self.rect.height))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.open:
                    dropdown_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                                self.rect.width, self.rect.height * self.visible_count)
                    if dropdown_rect.collidepoint(event.pos):
                        clicked_index = (event.pos[1] - self.rect.bottom) // self.rect.height + self.scroll_offset
                        if 0 <= clicked_index < len(self.options):
                            self.selected_option = clicked_index
                else:
                    if self.rect.collidepoint(event.pos):
                        self.open = True
                        self.scroll_offset = 0
            elif event.button == 4:  # Прокрутка колесиком вверх
                if self.open:
                    self.scroll_offset -= 1
                    self.scroll_offset = max(0, self.scroll_offset)
            elif event.button == 5:  # Прокрутка колесиком вниз
                if self.open:
                    self.scroll_offset += 1
                    self.scroll_offset = min(len(self.options) - self.visible_count, self.scroll_offset)

