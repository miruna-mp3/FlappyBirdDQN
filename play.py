import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import time
import argparse
import pygame

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


class PlayWrapper(FlappyBirdWrapper):
    """
    wrapper pentru play - afișează jocul cu pygame
    """

    def __init__(self, env, render_display=True, **kwargs):
        super().__init__(env, **kwargs)
        self.render_display = render_display
        self.screen = None

        # inițializează pygame pentru afișare (288 lățime x 512 înălțime)
        if render_display:
            pygame.init()
            self.screen = pygame.display.set_mode((288, 512))
            pygame.display.set_caption("Flappy Bird DQN")
            self.font = pygame.font.Font(None, 48)

        self.score = 0

    def _show_frame(self, rgb_frame, score=None):
        # afișează frame-ul curent în fereastră pygame
        if self.render_display and self.screen is not None:
            surface = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            self.screen.blit(surface, (0, 0))

            # afișează scorul pe ecran
            if score is not None:
                score_text = self.font.render(str(score), True, (255, 255, 255))
                score_rect = score_text.get_rect(center=(144, 50))
                self.screen.blit(score_text, score_rect)

            pygame.display.flip()

            # verifică dacă utilizatorul închide fereastra
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    exit()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        rgb_frame = self.env.render()
        self.score = 0

        self._show_frame(rgb_frame, self.score)
        frame = self._preprocess_frame(rgb_frame)

        # inițializează stiva de frame-uri
        for _ in range(self.stack_frames):
            self.frames.append(frame)

        return self._get_stacked_frames(), info

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        # execută acțiunea de mai multe ori (frame skip)
        for _ in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # actualizează scorul din info
            if 'score' in info:
                self.score = info['score']
            if terminated or truncated:
                break

        rgb_frame = self.env.render()
        self._show_frame(rgb_frame, self.score)
        frame = self._preprocess_frame(rgb_frame)
        self.frames.append(frame)

        # reward shaping
        if terminated or truncated:
            total_reward = -1.0
        else:
            total_reward += 0.1

        clipped_reward = np.clip(total_reward, -1.0, 1.0)
        return self._get_stacked_frames(), clipped_reward, terminated, truncated, info

    def close(self):
        if self.screen is not None:
            pygame.quit()
        super().close()


def play_episode(env, agent, max_steps=10000, show_qvalues=True):
    """
    rulează un singur episod și returnează reward-ul și lungimea
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        # selectează acțiunea greedy (fără explorare)
        action, q_values = agent.select_action(state, training=False)

        # execută acțiunea în mediu
        next_state, reward, terminated, truncated, info = env.step(action)
        score = info.get('score', 0)

        # afișează q-values și scor în timp real
        if show_qvalues:
            act_str = "SARI" if action == 1 else "STAI"
            print(f"\r  [{step:4d}]  Q0 {q_values[0]:7.3f}  Q1 {q_values[1]:7.3f}  [{act_str:4s}]  Score {score:3d}  R {episode_reward:7.2f}", end='', flush=True)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1
        state = next_state

        # încetinește pentru vizualizare
        time.sleep(0.02)

        if done:
            if show_qvalues:
                print()
            break

    return episode_reward, episode_length


def evaluate_agent(model_path, n_episodes=10, render=True, show_qvalues=True):
    """
    evaluează agentul pe mai multe episoade și afișează statistici
    """
    # creează mediul cu wrapper pentru afișare
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = PlayWrapper(env, render_display=render)

    # creează agentul și încarcă modelul antrenat
    agent = DQNAgent()
    try:
        agent.load(model_path)
        print()
        print(f"  [ MODEL ÎNCĂRCAT ]")
        print()
        print(f"  Fișier           {model_path}")
        print(f"  Pași antrenați   {agent.steps_done}")
        print(f"  Epsilon          {agent.epsilon:.4f}")
        print()
    except FileNotFoundError:
        print()
        print(f"  [ EROARE ]")
        print()
        print(f"  Modelul {model_path} nu există")
        print(f"  Rulează mai întâi: python train.py")
        print()
        env.close()
        return

    rewards = []
    lengths = []

    print(f"  Evaluare pe {n_episodes} episoade...")
    print()

    for episode in range(1, n_episodes + 1):
        print()
        print(f"  [ Episod {episode}/{n_episodes} ]")

        reward, length = play_episode(env, agent, show_qvalues=show_qvalues)
        rewards.append(reward)
        lengths.append(length)

        print(f"  Reward {reward:6.2f}  Lungime {length:4d}")

        if episode < n_episodes:
            time.sleep(0.5)

    # calculează și afișează statisticile finale
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    print()
    print(f"  [ STATISTICI ]")
    print()
    print(f"  Reward mediu     {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Reward maxim     {max_reward:.2f}")
    print(f"  Reward minim     {min_reward:.2f}")
    print(f"  Lungime medie    {np.mean(lengths):.0f}")
    print()

    env.close()
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluează agentul DQN pe Flappy Bird")
    parser.add_argument("--model", type=str, default="best_flappy_dqn.pth",
                        help="calea către model (implicit: best_flappy_dqn.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="numărul de episoade (implicit: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="fără randare vizuală")
    parser.add_argument("--no-qvalues", action="store_true",
                        help="fără afișare Q-values")

    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
        show_qvalues=not args.no_qvalues
    )
