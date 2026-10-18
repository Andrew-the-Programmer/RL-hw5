import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.network import QNetwork
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        online_net: QNetwork,
        target_net: QNetwork,
        replay_buffer: ReplayBuffer,
        gamma: float,
        learning_rate: float,
        batch_size: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        target_update_freq: int,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.online_net = online_net.to(self.device)
        self.target_net = target_net.to(self.device)

        _ = self.target_net.load_state_dict(self.online_net.state_dict())

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
        )

        self.learn_steps = 0

    def act(self, state, training: bool = True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.online_net.net[-1].out_features)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        current_q = self.online_net(states).gather(1, actions).squeeze(1)

        next_q = self.target_net(next_states).max(dim=1).values
        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1

        if self.learn_steps % self.target_update_freq == 0:
            _ = self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def step_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(
            path,
            map_location=self.device,
        )
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
