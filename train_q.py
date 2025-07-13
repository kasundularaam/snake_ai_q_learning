"""
Tabular Q-learning trainer for snake_env.SnakeEnv
Run headless for speed, then watch the trained agent.
"""

import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeEnv

# ---------- hyper-params ----------
EPISODES = 20_000
ALPHA = 0.1           # learning rate
GAMMA = 0.9           # discount factor
EPS_START = 1.0
EPS_END = 0.05
DECAY_RATE = 0.00025       # ε decay schedule
REPORT_EVERY = 1000

env = SnakeEnv()
q_table = np.zeros((128, 3))


def eps(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-DECAY_RATE * step)


scores, rolling, step = [], [], 0

for episode in range(1, EPISODES + 1):
    state = env.reset()
    done = False
    while not done:
        step += 1
        # ε-greedy
        if random.random() < eps(step):
            action = random.randrange(3)
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, done, _ = env.step(action)
        best_next = np.max(q_table[next_state])
        q_table[state, action] += ALPHA * \
            (reward + GAMMA * best_next - q_table[state, action])
        state = next_state

    scores.append(env.score)
    rolling.append(env.score)
    if episode % REPORT_EVERY == 0:
        print(
            f"Episode {episode:5d} | avg score last {REPORT_EVERY}: {np.mean(rolling):.2f}")
        rolling.clear()

# ---------- plot ----------
out_path = pathlib.Path("reward_curve.png")
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Snake – Tabular Q-learning")
plt.tight_layout()
plt.savefig(out_path)
print(f"Training finished ✔  (curve saved to {out_path})")

# ---------- watch the agent ----------
state = env.reset()
done = False
while not done:
    env.render()
    action = int(np.argmax(q_table[state]))      # greedy play
    state, _, done, _ = env.step(action)
env.render(delay_ms=1000)  # hold window for a sec
