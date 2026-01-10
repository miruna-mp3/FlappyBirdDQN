import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Agent DQN pentru Flappy Bird.
    Folosește policy network + target network și epsilon-greedy.
    """
    
    def __init__(
        self,
        n_actions=2,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100000,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=1000,
        device=None
    ):
        """
        Args:
            n_actions: Număr de acțiuni (2 pentru Flappy Bird)
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Epsilon inițial pentru exploration
            epsilon_end: Epsilon final
            epsilon_decay: Număr de steps pentru decay
            buffer_capacity: Capacitate replay buffer
            batch_size: Batch size pentru training
            target_update_freq: Frecvență de update a target network (în steps)
            device: 'cuda' sau 'cpu'
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Policy network (învață)
        self.policy_net = DQN(n_actions).to(self.device)
        
        # Target network (fixat periodic)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Întotdeauna în eval mode
        
        # Optimizer și loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Contor steps
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Selectează acțiune folosind epsilon-greedy.
        
        Args:
            state: State curent (4, 84, 84) numpy array
            training: Dacă True, folosește epsilon-greedy; altfel greedy
            
        Returns:
            action (int)
        """
        # Epsilon-greedy exploration
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Greedy action (Q-value maxim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def update_epsilon(self):
        """Actualizează epsilon (decay linear)"""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.steps_done / self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stochează tranziție în replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Un pas de antrenare DQN.
        Sample batch din replay buffer și actualizează policy network.
        
        Returns:
            loss (float) sau None dacă buffer-ul e prea mic
        """
        # Verifică dacă avem suficiente experiențe
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Conversie la tensori
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values curente (policy network)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-values pentru next state (target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss și backprop
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pentru stabilitate
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update counters
        self.steps_done += 1
        self.update_epsilon()
        
        # Update target network periodic
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """Salvează modelul"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Încarcă modelul"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']