import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from . import config as C

def init_pygame():
    pygame.display.init()
    pygame.font.init()
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    display = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    clock = pygame.time.Clock()
    return display, clock

def draw_game(game, display):
    display.fill((255,255,255))
    pygame.draw.rect(display, (255,0,0), pygame.Rect(game.apple[0], game.apple[1], C.CELL, C.CELL))
    for pos in game.body:
        pygame.draw.rect(display, (0,200,0), pygame.Rect(pos[0], pos[1], C.CELL, C.CELL))
