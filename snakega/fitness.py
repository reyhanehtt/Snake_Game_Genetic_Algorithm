import numpy as np
import random
from . import config as C
from .model import forward
from .game import Game

def _feature_vector(game: Game):
    _, is_front, is_left, is_right = game.blocked_directions()
    _, _, avn, svn = game.angle_with_apple()
    return np.array([is_left, is_front, is_right, avn[0], svn[0], avn[1], svn[1]], dtype=np.float32)[None, :]

def _predict_rel_turn(game: Game, genome_flat):
    probs = forward(_feature_vector(game), genome_flat)[0]
    return int(np.argmax(probs) - 1)  # {-1,0,1}

def _manhattan(a, b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def run_game_with_policy(genome_flat, headless=True, max_steps=None, rng_seed=None):
    # Rendering is imported lazily only when needed
    display = clock = None
    if not headless:
        from .render import init_pygame, draw_game
        import pygame
        display, clock = init_pygame()

    rng = random.Random(rng_seed if rng_seed is not None else 12345)
    game = Game(rng=rng)

    steps = max_steps or C.STEPS_PER_GAME
    score_dense = 0.0
    score_repeat = 0.0
    steps_since_apple = 0
    prev_turn = 0
    repeat_count = 0
    prev_dist = _manhattan(game.head, game.apple)

    for _ in range(steps):
        if not headless:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.alive = False

        rel_turn = _predict_rel_turn(game, genome_flat)
        repeat_count = repeat_count + 1 if rel_turn == prev_turn else 0
        prev_turn = rel_turn

        button = Game.button_from_relative_turn(game.body, rel_turn)
        game.step(button)

        if not game.alive:
            score_dense -= C.DEATH_PENALTY
            break

        steps_since_apple += 1
        if steps_since_apple > C.STARVE_STEPS:
            score_dense -= C.DEATH_PENALTY
            break

        dist = _manhattan(game.head, game.apple)
        score_dense += C.DIST_SHAPING * (prev_dist - dist)  # reward getting closer
        prev_dist = dist

        score_dense -= C.STEP_PENALTY  # small per-step tax

        if repeat_count > 8 and rel_turn != 0:
            score_repeat -= C.REPEAT_DIR_PENALTY
        else:
            score_repeat += C.REPEAT_DIR_BONUS

        if steps_since_apple == 1 and game.num_apples > 0:
            steps_since_apple = 0

        if not headless:
            from .render import draw_game
            import pygame
            draw_game(game, display)
            pygame.display.set_caption(f"SCORE: {game.score}")
            pygame.display.update()
            clock.tick(C.FPS_PLAY)

    fitness = score_dense + score_repeat + game.num_apples * C.APPLE_FITNESS_BONUS
    tie = 1000 * game.num_apples + game.num_moves
    return float(fitness), int(tie)
