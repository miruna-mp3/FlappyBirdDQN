import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from collections import deque
import time
import os
from datetime import datetime

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def train_dqn(
    n_episodes=5000,
    max_steps_per_episode=10000,
    save_freq=100,
    log_freq=10,
    save_path="flappy_dqn.pth"
):
    # creează mediul flappy bird cu preprocesare
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = FlappyBirdWrapper(env)

    # creează agentul dqn cu hiperparametrii optimizați
    agent = DQNAgent(
        n_actions=2,
        lr=3e-5,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.001,
        epsilon_decay=200000,
        buffer_capacity=50000,
        batch_size=32,
        target_update_freq=1000,
        tau=0.001,
        observation_steps=10000
    )

    # creează directorul pentru loguri
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/train_{timestamp}.csv"

    # inițializează fișierul csv pentru loguri
    with open(log_file, "w") as f:
        f.write("episode,reward,avg_reward,length,loss,epsilon,buffer_size,time\n")

    print()
    print("  [ FLAPPY BIRD DQN ]")
    print()
    print(f"  Dispozitiv       {agent.device}")
    print(f"  Episoade         {n_episodes}")
    print(f"  Fază observare   {agent.observation_steps} pași")
    print(f"  Fișier log       {log_file}")
    print()

    # variabile pentru tracking
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')

    start_time = time.time()

    # bucla principală de antrenare
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []

        # rulează un episod complet
        for step in range(max_steps_per_episode):
            # selectează acțiunea folosind epsilon-greedy
            action, _ = agent.select_action(state, training=True)

            # execută acțiunea în mediu
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # stochează tranziția în replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # actualizează rețeaua neuronală
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # actualizează statisticile
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)

        # afișează progresul la fiecare log_freq episoade
        if episode % log_freq == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(losses) if losses else 0
            elapsed = time.time() - start_time

            print(f"  [{episode:5d}]  R {episode_reward:6.2f}  Avg {avg_reward:6.2f}  Len {episode_length:4d}  Loss {avg_loss:.4f}  e {agent.epsilon:.3f}  Buf {len(agent.memory):6d}  T {elapsed:.0f}s")

            # scrie în fișierul csv
            with open(log_file, "a") as f:
                f.write(f"{episode},{episode_reward:.2f},{avg_reward:.2f},{episode_length},{avg_loss:.6f},{agent.epsilon:.4f},{len(agent.memory)},{elapsed:.0f}\n")

            # salvează cel mai bun model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(f"best_{save_path}")
                print(f"           [ NOU RECORD {best_avg_reward:.2f} ]")

        # salvare periodică
        if episode % save_freq == 0:
            agent.save(save_path)
            print(f"           [ Checkpoint salvat ]")

    # salvare finală
    agent.save(save_path)
    env.close()

    total_time = time.time() - start_time
    print()
    print("  [ ANTRENARE FINALIZATĂ ]")
    print()
    print(f"  Timp total       {total_time/60:.1f} minute")
    print(f"  Cel mai bun avg  {best_avg_reward:.2f}")
    print()

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    rewards, lengths = train_dqn(
        n_episodes=3000,
        max_steps_per_episode=10000,
        save_freq=100,
        log_freq=10,
        save_path="flappy_dqn.pth"
    )
