import flappy_bird_gymnasium
import gymnasium as gym

# Creează mediul
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Observation type: {type(obs)}")

# Rulează câteva frame-uri
for i in range(100):
    action = env.action_space.sample()  # acțiune random
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        print(f"Episode ended at step {i}")

env.close()
print("\n✓ Mediul funcționează corect!")