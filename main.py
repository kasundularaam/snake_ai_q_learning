import pygame
import numpy as np
import random
import pickle
import os
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 40
GRID_WIDTH = 10
GRID_HEIGHT = 10
GAME_AREA_WIDTH = GRID_WIDTH * GRID_SIZE  # 400px
GAME_AREA_HEIGHT = GRID_HEIGHT * GRID_SIZE  # 400px

# Full screen interface
WINDOW_WIDTH = 1200  # Much wider for UI
WINDOW_HEIGHT = 800  # Taller for better layout

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize snake at center
        center_x = GRID_WIDTH // 2
        center_y = GRID_HEIGHT // 2
        self.snake = [(center_x, center_y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.food = self._place_food()
        self.game_over = False
        self.steps_without_food = 0
        return self.get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH-1),
                    random.randint(0, GRID_HEIGHT-1))
            if food not in self.snake:
                return food

    def get_state(self):
        """Get current state representation for Q-learning"""
        if self.game_over:
            return None

        head = self.snake[0]

        # Danger detection (collision with wall or body)
        danger_up = (head[1] <= 0 or (head[0], head[1]-1) in self.snake)
        danger_down = (head[1] >= GRID_HEIGHT -
                       1 or (head[0], head[1]+1) in self.snake)
        danger_left = (head[0] <= 0 or (head[0]-1, head[1]) in self.snake)
        danger_right = (head[0] >= GRID_WIDTH -
                        1 or (head[0]+1, head[1]) in self.snake)

        # Food direction
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        # Current direction
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT

        state = (
            danger_up, danger_down, danger_left, danger_right,
            food_up, food_down, food_left, food_right,
            dir_up, dir_down, dir_left, dir_right
        )

        return state

    def step(self, action):
        """Execute one step in the game"""
        if self.game_over:
            return self.get_state(), 0, True

        # Map action to direction (0=straight, 1=right turn, 2=left turn)
        if action == 1:  # Turn right
            self.direction = Direction((self.direction.value + 1) % 4)
        elif action == 2:  # Turn left
            self.direction = Direction((self.direction.value - 1) % 4)
        # action == 0 means continue straight

        # Move snake
        head = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head[0] - 1, head[1])
        else:  # RIGHT
            new_head = (head[0] + 1, head[1])

        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
                new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.game_over = True
            return self.get_state(), -10, True

        # Check collision with self
        if new_head in self.snake:
            self.game_over = True
            return self.get_state(), -10, True

        self.snake.insert(0, new_head)

        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            self.steps_without_food += 1
            # Small penalty for not eating food
            reward = -0.1

        # Penalty for taking too long without eating
        if self.steps_without_food > 100:
            self.game_over = True
            reward = -5

        return self.get_state(), reward, self.game_over


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action values
        self.q_table = {}

        # Training statistics
        self.scores = []
        self.episodes = 0

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]  # 3 actions
        return self.q_table[state][action]

    def set_q_value(self, state, action, value):
        """Set Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        self.q_table[state][action] = value

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if state is None:
            return 0

        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action
        else:
            # Choose best action
            q_values = [self.get_q_value(state, a) for a in range(3)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        if state is None:
            return

        current_q = self.get_q_value(state, action)

        if next_state is None:
            # Terminal state
            next_q = 0
        else:
            # Best Q-value for next state
            next_q_values = [self.get_q_value(next_state, a) for a in range(3)]
            next_q = max(next_q_values)

        # Q-learning update
        new_q = current_q + self.alpha * \
            (reward + self.gamma * next_q - current_q)
        self.set_q_value(state, action, new_q)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename):
        """Load Q-table from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(self.q_table)} states")


class GameRenderer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Q-Learning Training Dashboard")
        self.clock = pygame.time.Clock()

        # Large, readable fonts
        self.font_title = pygame.font.Font(None, 64)
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Modern color scheme
        self.colors = {
            'background': (15, 15, 25),
            'game_bg': (25, 25, 35),
            'snake_head': (80, 255, 80),
            'snake_body': (50, 200, 50),
            'snake_border': (30, 150, 30),
            'food': (255, 80, 80),
            'food_glow': (255, 120, 120),
            'text_primary': (255, 255, 255),
            'text_secondary': (180, 180, 200),
            'text_accent': (100, 200, 255),
            'panel_bg': (30, 30, 45),
            'panel_border': (70, 70, 90),
            'progress_bg': (40, 40, 55),
            'progress_fill': (100, 200, 255),
            'grid_lines': (35, 35, 45)
        }

        # Layout positions
        self.game_x = 50
        self.game_y = (WINDOW_HEIGHT - GAME_AREA_HEIGHT) // 2
        self.stats_x = GAME_AREA_WIDTH + 100
        self.stats_y = 50

    def draw_text_with_background(self, text, font, color, bg_color, x, y, padding=10):
        """Draw text with background panel"""
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()

        # Background panel
        panel_rect = pygame.Rect(x - padding, y - padding,
                                 text_rect.width + padding * 2,
                                 text_rect.height + padding * 2)
        pygame.draw.rect(self.screen, bg_color, panel_rect)
        pygame.draw.rect(
            self.screen, self.colors['panel_border'], panel_rect, 2)

        # Text
        self.screen.blit(text_surface, (x, y))
        return text_rect.height + padding * 2

    def draw_progress_bar(self, x, y, width, height, progress, label=""):
        """Draw a modern progress bar"""
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.colors['progress_bg'], bg_rect)
        pygame.draw.rect(self.screen, self.colors['panel_border'], bg_rect, 2)

        # Fill
        if progress > 0:
            fill_width = int(width * min(progress, 1.0))
            fill_rect = pygame.Rect(x + 2, y + 2, fill_width - 4, height - 4)
            pygame.draw.rect(
                self.screen, self.colors['progress_fill'], fill_rect)

        # Label on bar
        if label:
            label_surface = self.font_small.render(
                label, True, self.colors['text_primary'])
            label_rect = label_surface.get_rect()
            label_x = x + (width - label_rect.width) // 2
            label_y = y + (height - label_rect.height) // 2
            self.screen.blit(label_surface, (label_x, label_y))

    def draw_game_area(self, game):
        """Draw the game area with snake and food"""
        # Game background
        game_rect = pygame.Rect(self.game_x, self.game_y,
                                GAME_AREA_WIDTH, GAME_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.colors['game_bg'], game_rect)
        pygame.draw.rect(
            self.screen, self.colors['panel_border'], game_rect, 3)

        # Grid lines
        for x in range(0, GAME_AREA_WIDTH + 1, GRID_SIZE):
            pygame.draw.line(self.screen, self.colors['grid_lines'],
                             (self.game_x + x, self.game_y),
                             (self.game_x + x, self.game_y + GAME_AREA_HEIGHT))
        for y in range(0, GAME_AREA_HEIGHT + 1, GRID_SIZE):
            pygame.draw.line(self.screen, self.colors['grid_lines'],
                             (self.game_x, self.game_y + y),
                             (self.game_x + GAME_AREA_WIDTH, self.game_y + y))

        # Snake
        for i, segment in enumerate(game.snake):
            rect = pygame.Rect(
                self.game_x + segment[0] * GRID_SIZE + 2,
                self.game_y + segment[1] * GRID_SIZE + 2,
                GRID_SIZE - 4, GRID_SIZE - 4
            )

            if i == 0:  # Head
                pygame.draw.rect(self.screen, self.colors['snake_head'], rect)
                pygame.draw.rect(
                    self.screen, self.colors['snake_border'], rect, 3)
            else:  # Body
                alpha = max(0.4, 1.0 - (i * 0.1))
                color = tuple(int(c * alpha)
                              for c in self.colors['snake_body'])
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(
                    self.screen, self.colors['snake_border'], rect, 2)

        # Food
        food_rect = pygame.Rect(
            self.game_x + game.food[0] * GRID_SIZE + 4,
            self.game_y + game.food[1] * GRID_SIZE + 4,
            GRID_SIZE - 8, GRID_SIZE - 8
        )
        pygame.draw.ellipse(self.screen, self.colors['food'], food_rect)

        # Food glow
        glow_rect = pygame.Rect(
            self.game_x + game.food[0] * GRID_SIZE,
            self.game_y + game.food[1] * GRID_SIZE,
            GRID_SIZE, GRID_SIZE
        )
        pygame.draw.rect(self.screen, self.colors['food_glow'], glow_rect, 4)

    def draw_stats_panel(self, game, agent, episode):
        """Draw the statistics panel"""
        x, y = self.stats_x, self.stats_y

        # Title
        title_text = self.font_title.render(
            "AI TRAINING", True, self.colors['text_accent'])
        self.screen.blit(title_text, (x, y))
        y += 80

        # Current Score
        score_height = self.draw_text_with_background(
            f"SCORE: {game.score}",
            self.font_large, self.colors['text_primary'],
            self.colors['panel_bg'], x, y, 15
        )
        y += score_height + 20

        # Episode
        episode_height = self.draw_text_with_background(
            f"Episode: {episode:,}",
            self.font_medium, self.colors['text_secondary'],
            self.colors['panel_bg'], x, y, 12
        )
        y += episode_height + 15

        # States learned
        states_height = self.draw_text_with_background(
            f"States: {len(agent.q_table):,}",
            self.font_medium, self.colors['text_secondary'],
            self.colors['panel_bg'], x, y, 12
        )
        y += states_height + 25

        # Exploration rate
        exploration_text = self.font_medium.render(
            "Exploration Rate:", True, self.colors['text_secondary'])
        self.screen.blit(exploration_text, (x, y))
        y += 40

        epsilon_text = self.font_large.render(
            f"{agent.epsilon:.3f}", True, self.colors['text_primary'])
        self.screen.blit(epsilon_text, (x, y))

        # Exploration progress bar
        self.draw_progress_bar(x + 150, y + 10, 200, 30,
                               agent.epsilon, f"{agent.epsilon:.1%}")
        y += 70

        # Performance metrics
        if len(agent.scores) > 0:
            recent_scores = agent.scores[-100:] if len(
                agent.scores) >= 100 else agent.scores
            avg_score = np.mean(recent_scores)
            max_score = max(agent.scores)

            # Average score
            avg_height = self.draw_text_with_background(
                f"Avg Score: {avg_score:.1f}",
                self.font_medium, self.colors['text_secondary'],
                self.colors['panel_bg'], x, y, 12
            )
            y += avg_height + 15

            # Best score
            best_height = self.draw_text_with_background(
                f"Best Score: {max_score}",
                self.font_medium, self.colors['text_accent'],
                self.colors['panel_bg'], x, y, 12
            )
            y += best_height + 25

            # Performance trend
            if len(agent.scores) >= 20:
                recent_avg = np.mean(agent.scores[-10:])
                older_avg = np.mean(agent.scores[-20:-10])

                if recent_avg > older_avg * 1.1:
                    trend_text = "↗ IMPROVING"
                    trend_color = (100, 255, 100)
                elif recent_avg < older_avg * 0.9:
                    trend_text = "↘ DECLINING"
                    trend_color = (255, 150, 100)
                else:
                    trend_text = "→ STABLE"
                    trend_color = self.colors['text_accent']

                trend_surface = self.font_medium.render(
                    trend_text, True, trend_color)
                self.screen.blit(trend_surface, (x, y))

    def draw_training_progress(self, episode):
        """Draw training progress at bottom"""
        y = WINDOW_HEIGHT - 120  # More space from bottom
        max_episodes = 1000
        progress = min(episode / max_episodes, 1.0)

        # Progress label
        progress_text = self.font_medium.render(
            "Training Progress:", True, self.colors['text_primary'])
        self.screen.blit(progress_text, (80, y))  # More margin from left

        # Progress bar with better spacing
        self.draw_progress_bar(320, y - 5, 450, 40,
                               progress, f"{episode:,} / {max_episodes:,}")

        # Percentage
        percent_text = self.font_medium.render(
            f"{progress:.1%}", True, self.colors['text_accent'])
        self.screen.blit(percent_text, (790, y))

    def draw_controls(self):
        """Draw control instructions"""
        y = WINDOW_HEIGHT - 60  # More space from bottom
        controls_text = "CONTROLS: SPACE = Pause/Resume • R = Reset • S = Save Model • ESC = Exit"
        controls_surface = self.font_small.render(
            controls_text, True, self.colors['text_secondary'])

        # Center the text
        text_rect = controls_surface.get_rect()
        x = (WINDOW_WIDTH - text_rect.width) // 2
        self.screen.blit(controls_surface, (x, y))

    def render(self, game, agent, episode):
        """Main render function"""
        # Clear screen
        self.screen.fill(self.colors['background'])

        # Draw all components
        self.draw_game_area(game)
        self.draw_stats_panel(game, agent, episode)
        self.draw_training_progress(episode)
        self.draw_controls()

        # Update display
        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return "pause"
                elif event.key == pygame.K_r:
                    return "reset"
                elif event.key == pygame.K_s:
                    return "save"
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True


def main():
    # Initialize game components
    game = SnakeGame()
    agent = QLearningAgent()
    renderer = GameRenderer()

    # Load existing model if available
    model_file = "snake_q_table.pkl"
    agent.load_model(model_file)

    # Training parameters
    episode = 0
    max_episodes = 10000
    render_every = 1  # Render every episode for smooth visuals
    save_every = 100   # Save model every N episodes

    running = True
    paused = False

    print("Starting Snake Q-Learning Training")
    print("Press SPACE to pause/unpause")
    print("Press ESC or close window to stop training")

    while running and episode < max_episodes:
        state = game.reset()
        total_reward = 0
        steps = 0

        while not game.game_over and running:
            # Handle events
            if episode % render_every == 0:
                event_result = renderer.handle_events()
                if event_result == False:
                    running = False
                    break
                elif event_result == "pause":
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event_result == "reset":
                    episode = 0
                    agent.scores = []
                    print("Training reset")
                elif event_result == "save":
                    agent.save_model(model_file)
                    print(f"Model saved manually at episode {episode}")

            if paused:
                renderer.clock.tick(10)
                continue

            # Agent chooses action
            action = agent.choose_action(state)

            # Execute action
            next_state, reward, done = game.step(action)
            total_reward += reward

            # Update Q-table
            agent.update(state, action, reward, next_state)

            state = next_state
            steps += 1

            # Render game
            if episode % render_every == 0:
                renderer.render(game, agent, episode)
                renderer.clock.tick(15)  # Smooth animation

        # Episode finished
        agent.scores.append(game.score)
        agent.episodes = episode
        agent.decay_epsilon()

        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(agent.scores[-100:]) if agent.scores else 0
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, States: {len(agent.q_table)}")

        # Save model periodically
        if episode % save_every == 0:
            agent.save_model(model_file)
            print(f"Model saved at episode {episode}")

        episode += 1

    # Final save
    agent.save_model(model_file)
    print(
        f"Training completed. Final model saved with {len(agent.q_table)} states")

    # Show final statistics
    if agent.scores:
        print(
            f"Final average score (last 100): {np.mean(agent.scores[-100:]):.2f}")
        print(f"Best score: {max(agent.scores)}")

    pygame.quit()


if __name__ == "__main__":
    main()
