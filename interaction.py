import pygame
import random

RES = 32
DIMS = (28, 28)
SCREEN = (RES * DIMS[0], RES * DIMS[1])
display = pygame.display.set_mode(SCREEN)

class Tile:

    def __init__(self, c, r) -> None:

        self.c = c
        self.r = r
        self.x = self.c * RES
        self.y = self.r * RES
        self.state = 0
        self.rect = pygame.Rect(self.x, self.y, RES, RES)

    def getColor(self):
        return pygame.Color(self.state, self.state, self.state)

    def draw(self):
        pygame.draw.rect(display, self.getColor(), self.rect)        