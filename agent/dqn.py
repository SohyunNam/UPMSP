import os
import collections
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.8
buffer_limit = 100000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        out = out.detach().numpy()
        coin = np.random.uniform(0, 1)
        if coin < epsilon:
            return np.random.randint(low=0, high=self.action_size)
        else:
            return np.argmax(out)


def train(q, q_target, memory, optimizer):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()