import random
from enum import Enum
from collections import namedtuple
import numpy as np
import pygame

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors (RGB)
BG_COLOR = (24, 28, 36)
GRID_COLOR = (35, 41, 52)
SNAKE_HEAD = (110, 231, 183)
SNAKE_BODY = (16, 185, 129)
SNAKE_BORDER = (6, 95, 70)
EYE_WHITE = (255, 255, 255)
EYE_PUPIL = (15, 23, 42)
FOOD_COLOR = (239, 68, 68)
TEXT_COLOR = (243, 244, 246)

BLOCK_SIZE = 42  # Slightly larger blocks so the 10x10 grid looks crisp and detailed

class SnakeEnv:
    def __init__(self, grid_size=10, use_distance_shaping=True, render_mode=True, fps=15):
        self.grid_size = grid_size
        self.max_cells = grid_size * grid_size
        self.use_distance_shaping = use_distance_shaping
        self.render_mode = render_mode
        self.fps = fps
        self.total_steps = 0

        if self.render_mode:
            pygame.init()
            self.width = self.grid_size * BLOCK_SIZE
            self.height = self.grid_size * BLOCK_SIZE + 45
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Master's RL Snake - Full Grid Goal")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 19, bold=True)

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        mid = self.grid_size // 2
        self.head = Point(mid, mid)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y)
        ]
        self.score = 0
        self.steps_since_food = 0
        self.food = None
        self._place_food()
        if self.render_mode:
            self._render_ui()
        return self.get_state()

    def _place_food(self):
        occupied = set(self.snake)
        free_cells = [
            Point(x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if Point(x, y) not in occupied
        ]
        if free_cells:
            self.food = random.choice(free_cells)
            return True
        else:
            self.food = None
            return False

    def _manhattan_distance(self, p1, p2):
        if p2 is None:
            return 0
        dx = min(abs(p1.x - p2.x), self.grid_size - abs(p1.x - p2.x))
        dy = min(abs(p1.y - p2.y), self.grid_size - abs(p1.y - p2.y))
        return dx + dy

    def step(self, action):
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self.total_steps += 1
        self.steps_since_food += 1

        prev_dist = self._manhattan_distance(self.head, self.food)
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.0
        done = False

        # 1. Check if food is eaten -> Grow by +1 permanently
        if self.food is not None and self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0

            # Check if the grid is now completely full with the snake!
            if len(self.snake) >= self.max_cells or not self._place_food():
                done = True
                reward = 50.0  # Victory bonus for filling 100% of the grid
        else:
            self.snake.pop()
            # Penalize overlapping body in RL reward, but DO NOT end the game
            if self.is_body_collision():
                reward = -5.0
            elif self.use_distance_shaping:
                curr_dist = self._manhattan_distance(self.head, self.food)
                reward = 0.3 if curr_dist < prev_dist else -0.35
            else:
                reward = -0.01

        if self.render_mode:
            self._render_ui()
            self.clock.tick(self.fps)

        return self.get_state(), reward, done, self.score

    def is_body_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return pt in self.snake[1:]

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x = (x + 1) % self.grid_size
        elif self.direction == Direction.LEFT:
            x = (x - 1) % self.grid_size
        elif self.direction == Direction.DOWN:
            y = (y + 1) % self.grid_size
        elif self.direction == Direction.UP:
            y = (y - 1) % self.grid_size
        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = Point((head.x - 1) % self.grid_size, head.y)
        point_r = Point((head.x + 1) % self.grid_size, head.y)
        point_u = Point(head.x, (head.y - 1) % self.grid_size)
        point_d = Point(head.x, (head.y + 1) % self.grid_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        food_x = self.food.x if self.food else head.x
        food_y = self.food.y if self.food else head.y

        state = [
            (dir_r and self.is_body_collision(point_r)) or (dir_l and self.is_body_collision(point_l)) or (dir_u and self.is_body_collision(point_u)) or (dir_d and self.is_body_collision(point_d)),
            (dir_u and self.is_body_collision(point_r)) or (dir_d and self.is_body_collision(point_l)) or (dir_l and self.is_body_collision(point_u)) or (dir_r and self.is_body_collision(point_d)),
            (dir_d and self.is_body_collision(point_r)) or (dir_u and self.is_body_collision(point_l)) or (dir_r and self.is_body_collision(point_u)) or (dir_l and self.is_body_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            food_x < head.x, food_x > head.x,
            food_y < head.y, food_y > head.y
        ]
        return np.array(state, dtype=np.float32)

    def _draw_eyes(self, head_x, head_y):
        eye_radius = 5
        pupil_radius = 2
        offset_near = 11
        offset_far = BLOCK_SIZE - 11

        if self.direction == Direction.RIGHT:
            eyes = [(head_x + offset_far, head_y + offset_near), (head_x + offset_far, head_y + offset_far)]
            pupil_shift = (2, 0)
        elif self.direction == Direction.LEFT:
            eyes = [(head_x + offset_near, head_y + offset_near), (head_x + offset_near, head_y + offset_far)]
            pupil_shift = (-2, 0)
        elif self.direction == Direction.UP:
            eyes = [(head_x + offset_near, head_y + offset_near), (head_x + offset_far, head_y + offset_near)]
            pupil_shift = (0, -2)
        else:  # DOWN
            eyes = [(head_x + offset_near, head_y + offset_far), (head_x + offset_far, head_y + offset_far)]
            pupil_shift = (0, 2)

        for ex, ey in eyes:
            pygame.draw.circle(self.display, EYE_WHITE, (ex, ey), eye_radius)
            pygame.draw.circle(self.display, EYE_PUPIL, (ex + pupil_shift[0], ey + pupil_shift[1]), pupil_radius)

    def _render_ui(self):
        self.display.fill(BG_COLOR)
        top_offset = 45

        for x in range(0, self.width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, top_offset), (x, self.height))
        for y in range(top_offset, self.height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.width, y))

        # Draw body segments first, then head on top so eyes stay clearly visible
        for idx in range(len(self.snake) - 1, -1, -1):
            pt = self.snake[idx]
            px = pt.x * BLOCK_SIZE
            py = pt.y * BLOCK_SIZE + top_offset
            outer_rect = pygame.Rect(px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            inner_rect = pygame.Rect(px + 4, py + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8)
            pygame.draw.rect(self.display, SNAKE_BORDER, outer_rect, border_radius=7)
            color = SNAKE_HEAD if idx == 0 else SNAKE_BODY
            pygame.draw.rect(self.display, color, inner_rect, border_radius=5)

            if idx == 0:
                self._draw_eyes(px, py)

        # Draw Food if grid is not yet completely full
        if self.food is not None:
            food_center = (self.food.x * BLOCK_SIZE + BLOCK_SIZE // 2, self.food.y * BLOCK_SIZE + top_offset + BLOCK_SIZE // 2)
            pygame.draw.circle(self.display, FOOD_COLOR, food_center, BLOCK_SIZE // 2 - 5)

        fill_pct = (len(self.snake) / self.max_cells) * 100
        header_text = self.font.render(
            f"Length: {len(self.snake)}/{self.max_cells} ({fill_pct:.0f}%) | Food: {self.score}",
            True,
            TEXT_COLOR
        )
        self.display.blit(header_text, (12, 10))
        pygame.display.flip()