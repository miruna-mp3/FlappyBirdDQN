import gymnasium as gym
import flappy_bird_gymnasium
from env_wrapper import FlappyBirdWrapper

# Creează mediul cu wrapper
env = gym.make("FlappyBird-v0")
env = FlappyBirdWrapper(env)

# Test reset
obs, info = env.reset()
print(f"✓ Reset OK")
print(f"  Shape: {obs.shape}")
print(f"  Type: {obs.dtype}")
print(f"  Min: {obs.min():.3f}, Max: {obs.max():.3f}")

# Test step
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i == 0:
        print(f"\n✓ Step OK")
        print(f"  Shape: {obs.shape}")
        print(f"  Reward: {reward}")

# Verificare finală
assert obs.shape == (4, 84, 84), f"Expected (4, 84, 84), got {obs.shape}"
assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
assert 0 <= obs.min() <= 1, f"Values out of range [0, 1]"
assert 0 <= obs.max() <= 1, f"Values out of range [0, 1]"

print("\n✅ Toate verificările au trecut!")
print(f"   Input final pentru CNN: {obs.shape}")

env.close()