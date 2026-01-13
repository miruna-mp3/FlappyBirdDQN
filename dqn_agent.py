import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    agent dqn cu policy network, target network și epsilon-greedy
    """

    def __init__(
        self,
        n_actions=2,
        lr=3e-5,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=0.0001,
        epsilon_decay=500000,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=1000,
        tau=0.001,
        observation_steps=10000,
        device=None
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.observation_steps = observation_steps

        # selectează dispozitivul (GPU dacă e disponibil)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # rețeaua principală (policy) și rețeaua țintă (target)
        self.policy_net = DQN(n_actions).to(self.device)
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizator și funcție de loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        # buffer pentru experience replay
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.steps_done = 0

    def select_action(self, state, training=True):
        # selectează acțiune folosind epsilon-greedy biased, calculează q-values pentru starea curentă
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            q_values_list = q_values.cpu().numpy()[0].tolist()

        # în faza de observare sau cu probabilitate epsilon: acțiune aleatoare
        if training and (self.steps_done < self.observation_steps or np.random.rand() < self.epsilon):
            # preferință pentru "nu sări" - evită lovirea tavanului
            if np.random.rand() < 0.15:
                action = 1  # sari
            else:
                action = 0  # nu sări
            return action, q_values_list
        else:
            # acțiune greedy - alege acțiunea cu q-value maxim
            action = q_values.argmax(dim=1).item()
            return action, q_values_list

    def update_epsilon(self):
        # decay liniar al epsilon-ului
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.steps_done / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
        )

    def store_transition(self, state, action, reward, next_state, done):
        # adaugă tranziția în replay buffer
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        # în faza de observare doar colectează experiențe, nu antrenează
        if self.steps_done < self.observation_steps:
            self.steps_done += 1
            return None

        # verifică dacă avem suficiente experiențe în buffer
        if len(self.memory) < self.batch_size:
            return None

        # extrage un batch aleator din replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # convertește la tensori pytorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # calculează q-values curente
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # calculează q-values țintă folosind target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # calculează loss-ul și face backpropagation
        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # clip gradienți pentru stabilitate
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.steps_done += 1
        self.update_epsilon()

        # soft update pentru target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

        return loss.item()

    def save(self, path):
        # salvează modelul și starea antrenării
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        # încarcă modelul și starea antrenării
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
