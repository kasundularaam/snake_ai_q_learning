"""
Minimal Snake environment with optional pygame rendering
Designed for tabular Q-learning (128 states × 3 actions)
"""

import random
import sys
import numpy as np
import pygame                      # only needed when you call render()

# ---------- board & actions ----------
BOARD_W = BOARD_H = 10             # change here for larger boards
CELL = 30                       # pixel size when rendering
FPS = 15                       # pygame framerate

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_VECS = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
# action indices: 0 = straight, 1 = turn right, 2 = turn left
TURN = {0: 0, 1: 1, 2: -1}


class SnakeEnv:
    """Gym-style API: reset(), step(action) -> (state, reward, done, info)"""

    def reset(self):
        self.snake = [(BOARD_W // 2, BOARD_H // 2)]
        self.dir = RIGHT
        self.score = 0
        self._spawn_food()
        return self._state()

    # -------- core transition ----------
    def step(self, action: int):
        self.dir = (self.dir + TURN[action]) % 4
        head_x, head_y = self.snake[0]
        dx, dy = DIR_VECS[self.dir]
        new_head = (head_x + dx, head_y + dy)

        reward, done = 0, False
        if (not 0 <= new_head[0] < BOARD_W) or (not 0 <= new_head[1] < BOARD_H) \
           or new_head in self.snake:
            reward, done = -1, True               # hit wall or self
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward, self.score = +1, self.score + 1
                self._spawn_food()
            else:
                self.snake.pop()

        return self._state(), reward, done, {}

    # -------- helpers ----------
    def _spawn_food(self):
        empty = [(x, y) for x in range(BOARD_W) for y in range(BOARD_H)
                 if (x, y) not in self.snake]
        self.food = random.choice(empty)

    def _danger(self, dirn):
        head_x, head_y = self.snake[0]
        dx, dy = DIR_VECS[dirn]
        nx, ny = head_x + dx, head_y + dy
        return (nx, ny) in self.snake or not (0 <= nx < BOARD_W) or not (0 <= ny < BOARD_H)

    def _state(self):
        head_x, head_y = self.snake[0]
        dir_l, dir_r = (self.dir - 1) % 4, (self.dir + 1) % 4

        food_left = self.food[0] < head_x
        food_up = self.food[1] < head_y

        danger_straight = self._danger(self.dir)
        danger_right = self._danger(dir_r)
        danger_left = self._danger(dir_l)

        # pack bits: start with 2-bit direction (0-3), then shift in the 5 booleans
        code = self.dir                      # 2 bits
        for b in (food_left, food_up,
                  danger_straight, danger_right, danger_left):
            code = (code << 1) | int(b)
        return code                          # guaranteed 0-127

    # -------- optional pygame render ----------

    def render(self, delay_ms=0):
        if not hasattr(self, "_pg_init"):
            pygame.init()
            self._screen = pygame.display.set_mode(
                (BOARD_W * CELL, BOARD_H * CELL))
            pygame.display.set_caption("Snake – Tabular Q")
            self._clock = pygame.time.Clock()
            self._pg_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self._screen.fill((30, 30, 30))

        # food
        fx, fy = self.food
        pygame.draw.rect(self._screen, (200, 40, 40),
                         pygame.Rect(fx * CELL, fy * CELL, CELL, CELL))

        # snake
        for i, (x, y) in enumerate(self.snake):
            color = (60, 220, 60) if i else (120, 255, 120)
            pygame.draw.rect(self._screen, color,
                             pygame.Rect(x * CELL, y * CELL, CELL, CELL))

        pygame.display.flip()
        self._clock.tick(FPS)
        if delay_ms:
            pygame.time.wait(delay_ms)
