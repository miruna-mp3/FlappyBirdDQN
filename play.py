import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import time
import argparse

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def play_episode(env, agent, render=True, max_steps=10000):
    """
    Rulează un episod cu agentul antrenat.
    
    Args:
        env: Mediul
        agent: Agentul DQN
        render: Dacă True, randează vizual
        max_steps: Steps maxime per episod
        
    Returns:
        episode_reward, episode_length
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in range(max_steps):
        # Selectează acțiune (greedy, fără exploration)
        action = agent.select_action(state, training=False)
        
        # Execută acțiune
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        if render:
            time.sleep(0.01)  # Slow down pentru vizualizare
        
        if done:
            break
    
    return episode_reward, episode_length


def evaluate_agent(model_path, n_episodes=10, render=True):
    """
    Evaluează agentul pe mai multe episoade.
    
    Args:
        model_path: Path la modelul antrenat
        n_episodes: Număr de episoade de test
        render: Dacă True, randează vizual
    """
    # Creează mediu
    if render:
        env = gym.make("FlappyBird-v0", render_mode="human")
    else:
        env = gym.make("FlappyBird-v0")
    env = FlappyBirdWrapper(env)
    
    # Creează și încarcă agent
    agent = DQNAgent()
    try:
        agent.load(model_path)
        print(f"✅ Model încărcat: {model_path}")
        print(f"   Steps done: {agent.steps_done}")
        print(f"   Epsilon: {agent.epsilon:.4f}\n")
    except FileNotFoundError:
        print(f"❌ Modelul {model_path} nu există!")
        print(f"   Rulează mai întâi: python train.py")
        env.close()
        return
    
    # Rulează episoade
    rewards = []
    lengths = []
    
    print(f"🎮 Evaluare pe {n_episodes} episoade...\n")
    
    for episode in range(1, n_episodes + 1):
        reward, length = play_episode(env, agent, render=render)
        rewards.append(reward)
        lengths.append(length)
        
        print(f"Episode {episode:2d} | Reward: {reward:6.2f} | Length: {length:4d}")
        
        if render and episode < n_episodes:
            time.sleep(0.5)  # Pauză între episoade
    
    # Statistici
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    print(f"\n📊 Statistici:")
    print(f"   Reward mediu:  {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   Reward max:    {max_reward:.2f}")
    print(f"   Reward min:    {min_reward:.2f}")
    print(f"   Length mediu:  {np.mean(lengths):.0f}")
    
    env.close()
    
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluează agent DQN pe Flappy Bird")
    parser.add_argument("--model", type=str, default="best_flappy_dqn.pth",
                        help="Path la model (default: best_flappy_dqn.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Număr de episoade (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Fără randare vizuală")
    
    args = parser.parse_args()
    
    # Evaluare
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render
    )