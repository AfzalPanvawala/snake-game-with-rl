import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LinearQNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, use_double_dqn=True):
        self.n_games = 0
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.use_double_dqn = use_double_dqn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=50_000)

        self.policy_net = LinearQNet().to(self.device)
        self.target_net = LinearQNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, eval_mode=False):
        final_move = [0, 0, 0]
        if not eval_mode and random.random() < self.epsilon:
            final_move[random.randint(0, 2)] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                prediction = self.policy_net(state_tensor)
            final_move[torch.argmax(prediction).item()] = 1
        return final_move

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        action_indices = torch.argmax(actions, dim=1).unsqueeze(1)
        current_q = self.policy_net(states).gather(1, action_indices).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                best_next_actions = torch.argmax(self.policy_net(next_states), dim=1).unsqueeze(1)
                next_q = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (~dones))

        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.n_games += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.n_games % 10 == 0:
            self.update_target_network()

    def save_checkpoint(self, filepath="snake_dqn.pth"):
        torch.save({'model_state_dict': self.policy_net.state_dict(), 'n_games': self.n_games}, filepath)