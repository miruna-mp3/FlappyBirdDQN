import numpy as np
from replay_buffer import ReplayBuffer

# Creează buffer
buffer = ReplayBuffer(capacity=1000)
print(f"✓ Buffer creat cu capacitate 1000")
print(f"  Lungime inițială: {len(buffer)}")

# Adaugă tranziții dummy
print("\n✓ Adăugare tranziții...")
for i in range(150):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = np.random.randint(0, 2)
    reward = np.random.rand()
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = np.random.rand() > 0.9  # 10% șansă de episod terminat
    
    buffer.push(state, action, reward, next_state, done)

print(f"  Lungime după adăugare: {len(buffer)}")

# Sample un batch
print("\n✓ Sampling batch de 32...")
batch_size = 32
states, actions, rewards, next_states, dones = buffer.sample(batch_size)

print(f"  States shape: {states.shape}")
print(f"  Actions shape: {actions.shape}")
print(f"  Rewards shape: {rewards.shape}")
print(f"  Next states shape: {next_states.shape}")
print(f"  Dones shape: {dones.shape}")

# Verificări
assert states.shape == (batch_size, 4, 84, 84), f"Expected (32, 4, 84, 84), got {states.shape}"
assert actions.shape == (batch_size,), f"Expected (32,), got {actions.shape}"
assert rewards.shape == (batch_size,), f"Expected (32,), got {rewards.shape}"
assert next_states.shape == (batch_size, 4, 84, 84), f"Expected (32, 4, 84, 84), got {next_states.shape}"
assert dones.shape == (batch_size,), f"Expected (32,), got {dones.shape}"

# Verificare tipuri
assert states.dtype == np.float32
assert actions.dtype == np.int64
assert rewards.dtype == np.float32
assert next_states.dtype == np.float32
assert dones.dtype == np.float32

print("\n✓ Verificare sampling multiplu...")
for _ in range(5):
    batch = buffer.sample(32)
    assert len(batch) == 5  # 5 componente

# Test capacitate maximă
print("\n✓ Test capacitate maximă...")
large_buffer = ReplayBuffer(capacity=100)
for i in range(150):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    large_buffer.push(state, 0, 0.0, state, False)

print(f"  După 150 push-uri în buffer cu cap=100: len={len(large_buffer)}")
assert len(large_buffer) == 100, "Buffer ar trebui să mențină doar ultimele 100"

print("\n✅ Toate verificările au trecut!")
print("   Replay Buffer funcționează corect")