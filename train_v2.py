import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from collections import deque
import time
import os
from datetime import datetime

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def log(msg, file=None):
    # afișează în terminal și scrie în fișier
    print(msg)
    if file:
        file.write(msg + "\n")
        file.flush()


def train_v2(n_episodes=3000, save_path="flappy_dqn.pth"):
    # creează mediul cu preprocesare
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = FlappyBirdWrapper(env, frame_skip=2)

    # creează agentul cu hiperparametrii optimizați
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
    log_path = f"logs/train_v2_{timestamp}.txt"
    log_file = open(log_path, "w")

    log("", log_file)
    log("  [ FLAPPY BIRD DQN ]", log_file)
    log("", log_file)
    log(f"  Dispozitiv       {agent.device}", log_file)
    log(f"  Episoade         {n_episodes}", log_file)
    log(f"  Fază observare   {agent.observation_steps} pași", log_file)
    log(f"  Log              {log_path}", log_file)
    log("", log_file)

    # variabile pentru tracking
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_scores = deque(maxlen=100)
    best_avg = -float('inf')
    best_avg_score = 0
    start_time = time.time()

    # bucla principală de antrenare
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_score = 0

        # rulează un episod complet
        for step in range(10000):
            # selectează acțiunea
            action, _ = agent.select_action(state, training=True)

            # execută acțiunea
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # actualizează scorul (tuburi trecute)
            if 'score' in info:
                episode_score = info['score']

            # stochează și antrenează
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # actualizează statisticile
        recent_rewards.append(episode_reward)
        recent_lengths.append(episode_length)
        recent_scores.append(episode_score)

        # afișează progresul
        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_score = np.mean(recent_scores)
            elapsed = time.time() - start_time

            # arată dacă suntem în faza de observare sau antrenare
            phase = "[OBS]" if agent.steps_done < agent.observation_steps else "[TRN]"
            log(f"  [{episode:4d}] {phase}  [Score] {episode_score:3d}  [AvgS] {avg_score:5.1f}  [R] {episode_reward:6.1f}  [AvgR] {avg_reward:6.1f}  [ε] {agent.epsilon:.4f}  [T] {elapsed/60:.1f}m", log_file)

            # salvează cel mai bun model
            if avg_reward > best_avg and episode > 100:
                best_avg = avg_reward
                best_avg_score = avg_score
                agent.save(f"best_{save_path}")
                log(f"          [ NOU RECORD {best_avg:.1f} | Score {best_avg_score:.1f} ]", log_file)

        # salvare periodică
        if episode % 200 == 0:
            agent.save(save_path)
            log(f"          [ Checkpoint salvat ]", log_file)

    # salvare finală
    agent.save(save_path)
    env.close()

    log("", log_file)
    log("  [ ANTRENARE FINALIZATĂ ]", log_file)
    log("", log_file)
    log(f"  Timp total       {(time.time() - start_time)/60:.1f} minute", log_file)
    log(f"  Cel mai bun avg  {best_avg:.1f}", log_file)
    log(f"  Cel mai bun scor {best_avg_score:.1f}", log_file)
    log("", log_file)

    log_file.close()


if __name__ == "__main__":
    train_v2(n_episodes=3000)
