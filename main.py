import pygame
import numpy as np
import random
import pickle
import os
from collections import deque
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 15
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE + 100  # Extra space for UI

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
        pygame.display.set_caption("Snake Game - Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def render(self, game, agent, episode):
        """Render the game state"""
        self.screen.fill(BLACK)

        # Draw snake
        for segment in game.snake:
            rect = pygame.Rect(segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                               GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)
            pygame.draw.rect(self.screen, WHITE, rect, 1)

        # Draw food
        food_rect = pygame.Rect(game.food[0] * GRID_SIZE, game.food[1] * GRID_SIZE,
                                GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, RED, food_rect)

        # Draw UI
        y_offset = GRID_HEIGHT * GRID_SIZE + 10

        score_text = self.font.render(f"Score: {game.score}", True, WHITE)
        self.screen.blit(score_text, (10, y_offset))

        episode_text = self.font.render(f"Episode: {episode}", True, WHITE)
        self.screen.blit(episode_text, (200, y_offset))

        epsilon_text = self.font.render(
            f"Epsilon: {agent.epsilon:.3f}", True, WHITE)
        self.screen.blit(epsilon_text, (400, y_offset))

        states_text = self.font.render(
            f"States: {len(agent.q_table)}", True, WHITE)
        self.screen.blit(states_text, (10, y_offset + 30))

        if len(agent.scores) > 0:
            avg_score = np.mean(agent.scores[-100:])
            avg_text = self.font.render(
                f"Avg Score (last 100): {avg_score:.1f}", True, WHITE)
            self.screen.blit(avg_text, (250, y_offset + 30))

        pygame.display.flip()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return "pause"
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
    render_every = 10  # Render every N episodes
    save_every = 100   # Save model every N episodes

    running = True
    paused = False

    print("Starting Snake Q-Learning Training")
    print("Press SPACE to pause/unpause")
    print("Close window to stop training")

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
                renderer.clock.tick(10)  # Control game speed

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
