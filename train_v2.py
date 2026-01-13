import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from collections import deque
import time

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


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

    print()
    print("  [ FLAPPY BIRD DQN ]")
    print()
    print(f"  Dispozitiv       {agent.device}")
    print(f"  Episoade         {n_episodes}")
    print(f"  Fază observare   {agent.observation_steps} pași")
    print()

    # variabile pentru tracking
    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    best_avg = -float('inf')
    start_time = time.time()

    # bucla principală de antrenare
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        # rulează un episod complet
        for step in range(10000):
            # selectează acțiunea
            action, _ = agent.select_action(state, training=True)

            # execută acțiunea
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

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

        # afișează progresul
        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            elapsed = time.time() - start_time

            print(f"  [{episode:4d}]  [Reward] {episode_reward:6.1f}  [Avg] {avg_reward:6.1f}  [Len] {episode_length:4d}  [ε] {agent.epsilon:.4f}  [Buffer] {len(agent.memory):5d}  [T] {elapsed/60:.1f}m")

            # salvează cel mai bun model
            if avg_reward > best_avg and episode > 100:
                best_avg = avg_reward
                agent.save(f"best_{save_path}")
                print(f"          [ NOU RECORD {best_avg:.1f} ]")

        # salvare periodică
        if episode % 200 == 0:
            agent.save(save_path)
            print(f"          [ Checkpoint salvat ]")

    # salvare finală
    agent.save(save_path)
    env.close()

    print()
    print("  [ ANTRENARE FINALIZATĂ ]")
    print()
    print(f"  Timp total       {(time.time() - start_time)/60:.1f} minute")
    print(f"  Cel mai bun avg  {best_avg:.1f}")
    print()


if __name__ == "__main__":
    train_v2(n_episodes=3000)
