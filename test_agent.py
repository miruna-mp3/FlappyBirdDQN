import numpy as np
import torch
from dqn_agent import DQNAgent

print("✓ Creare agent...")
agent = DQNAgent(
    n_actions=2,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=10000,
    buffer_capacity=1000,
    batch_size=32,
    target_update_freq=100
)

print(f"  Device: {agent.device}")
print(f"  Epsilon inițial: {agent.epsilon}")
print(f"  Replay buffer size: {len(agent.memory)}")

# Test selectare acțiune
print("\n✓ Test selectare acțiune...")
state = np.random.rand(4, 84, 84).astype(np.float32)

# Acțiune cu exploration
action = agent.select_action(state, training=True)
print(f"  Acțiune (training mode): {action}")
assert action in [0, 1], "Acțiune invalidă"

# Acțiune greedy
action = agent.select_action(state, training=False)
print(f"  Acțiune (eval mode): {action}")
assert action in [0, 1], "Acțiune invalidă"

# Test stocare tranziții
print("\n✓ Adăugare tranziții în buffer...")
for i in range(100):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = np.random.randint(0, 2)
    reward = np.random.rand()
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = False
    agent.store_transition(state, action, reward, next_state, done)

print(f"  Buffer size după adăugare: {len(agent.memory)}")
assert len(agent.memory) == 100, "Buffer size incorect"

# Test training step
print("\n✓ Test training step...")
loss = agent.train_step()
print(f"  Loss primul step: {loss:.4f}")
assert loss is not None, "Loss ar trebui să existe când avem suficiente samples"
assert isinstance(loss, float), "Loss ar trebui să fie float"

# Mai multe training steps
print("\n✓ Test training multiplu (50 steps)...")
losses = []
for _ in range(50):
    loss = agent.train_step()
    if loss is not None:
        losses.append(loss)

print(f"  Steps efectuați: {len(losses)}")
print(f"  Loss mediu: {np.mean(losses):.4f}")
print(f"  Steps done: {agent.steps_done}")
print(f"  Epsilon după decay: {agent.epsilon:.4f}")

# Verificare epsilon decay
assert agent.epsilon < agent.epsilon_start, "Epsilon ar trebui să scadă"

# Test save/load
print("\n✓ Test save/load...")
agent.save("test_model.pth")
print("  Model salvat")

# Creează agent nou și încarcă
agent2 = DQNAgent()
agent2.load("test_model.pth")
print("  Model încărcat")
print(f"  Steps done după load: {agent2.steps_done}")
print(f"  Epsilon după load: {agent2.epsilon:.4f}")

assert agent2.steps_done == agent.steps_done, "Steps done nu s-a păstrat"

# Cleanup
import os
os.remove("test_model.pth")
print("  Test file șters")

print("\n✅ Toate verificările au trecut!")
print("   Agent DQN funcționează corect")