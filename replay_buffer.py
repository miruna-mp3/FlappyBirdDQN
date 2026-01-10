import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer pentru DQN.
    Stochează tranziții (s, a, r, s', done) și oferă sampling aleator.
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Numărul maxim de tranziții stocate
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Adaugă o tranziție în buffer.
        
        Args:
            state: State curent (4, 84, 84)
            action: Acțiune (int)
            reward: Reward (float)
            next_state: State următor (4, 84, 84)
            done: Episode terminat (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample aleator un batch de tranziții.
        
        Args:
            batch_size: Numărul de tranziții de returnat
            
        Returns:
            Tuple de numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Sample aleator
        batch = random.sample(self.buffer, batch_size)
        
        # Separă componentele
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Conversie la numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Returnează numărul curent de tranziții în buffer"""
        return len(self.buffer)