import math, random
import numpy as np
from . import config as C

class Game:
    def __init__(self, rng=None):
        self.rng = rng or random.Random(123)
        self.reset()

    def reset(self):
        self.head = [C.CELL*10, C.CELL*10]
        self.body = [[C.CELL*10, C.CELL*10], [C.CELL*9, C.CELL*10], [C.CELL*8, C.CELL*10]]
        self.apple = self._spawn_apple()
        self.score = 0
        self.alive = True
        self.num_apples = 0
        self.num_moves = 0

    def _spawn_apple(self):
        while True:
            pos = [self.rng.randrange(1, C.GRID_SIZE) * C.CELL,
                   self.rng.randrange(1, C.GRID_SIZE) * C.CELL]
            if pos not in self.body:
                return pos

    def _collision(self, head):
        return (
            head[0] >= C.WIDTH or head[1] >= C.HEIGHT or head[0] < 0 or head[1] < 0 or
            head in self.body[1:]
        )

    def blocked_directions(self):
        cur = np.array(self.body[0]) - np.array(self.body[1])
        left = np.array([cur[1], -cur[0]])
        right = np.array([-cur[1], cur[0]])
        def blocked(vec):
            nxt = (np.array(self.body[0]) + vec).tolist()
            return 1 if self._collision(nxt) else 0
        return cur, blocked(cur), blocked(left), blocked(right)

    def angle_with_apple(self):
        apple_vec = np.array(self.apple) - np.array(self.body[0])
        snake_vec = np.array(self.body[0]) - np.array(self.body[1])
        na = np.linalg.norm(apple_vec) or 1.0
        ns = np.linalg.norm(snake_vec) or 1.0
        avn = apple_vec/na
        svn = snake_vec/ns
        angle = math.atan2(avn[1]*svn[0]-avn[0]*svn[1], avn[1]*svn[1]+avn[0]*svn[0]) / math.pi
        return angle, snake_vec, avn, svn

    @staticmethod
    def button_from_relative_turn(body, rel_turn):
        cur = np.array(body[0]) - np.array(body[1])
        new_dir = cur.copy()
        if rel_turn == -1:
            new_dir = np.array([cur[1], -cur[0]])
        elif rel_turn == 1:
            new_dir = np.array([-cur[1], cur[0]])
        if new_dir.tolist() == [C.CELL, 0]:
            return 1
        if new_dir.tolist() == [-C.CELL, 0]:
            return 0
        if new_dir.tolist() == [0, C.CELL]:
            return 2
        return 3

    def step(self, button):
        if button == 1:       # right
            self.head[0] += C.CELL
        elif button == 0:     # left
            self.head[0] -= C.CELL
        elif button == 2:     # down
            self.head[1] += C.CELL
        elif button == 3:     # up
            self.head[1] -= C.CELL

        if self.head == self.apple:
            self.num_apples += 1
            self.score += C.APPLE_BONUS_SCORE
            self.body.insert(0, list(self.head))
            self.apple = self._spawn_apple()
        else:
            self.body.insert(0, list(self.head))
            self.body.pop()

        if self._collision(self.head):
            self.alive = False

        self.num_moves += 1
