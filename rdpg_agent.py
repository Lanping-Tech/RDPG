import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import deque

from util import *

import numpy as np

import random

USE_CUDA = False #torch.cuda.is_available()
device = torch.device('cpu') #torch.device('cuda' if USE_CUDA else 'cpu')
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class ActorNet(nn.Module):
    def __init__(self, n_features, n_a_hidden):
        super(ActorNet,self).__init__()

        self.fc1 = nn.Linear(n_features, n_a_hidden)
        self.fc1.weight.data.normal_(0,0.1)

        self.fc2 = nn.Linear(n_a_hidden, 1)
        self.fc2.weight.data.normal_(0,0.1)

    def forward(self,x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class CriticNet(nn.Module):
    def __init__(self, n_c_hidden):
        super(CriticNet,self).__init__()

        self.n_c_hidden = n_c_hidden

        self.rnn = nn.RNN(1, n_c_hidden, batch_first=True)

        self.fc1 = nn.Linear(1, n_c_hidden)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(2 * n_c_hidden, n_c_hidden)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(n_c_hidden, 1)
        self.fc3.weight.data.normal_(0, 0.1)

    
    def forward(self, xs):
        ts, a = xs

        ts = ts.unsqueeze(-1)
        hx = torch.zeros(1, ts.shape[0], self.n_c_hidden)

        ts_out, hx = self.rnn(ts, hx)
        ts_out = ts_out[:, -1, :]

        a_out = self.fc1(a)

        concat_out = torch.cat([a_out, ts_out], dim=-1)

        output = self.fc2(concat_out)
        output = F.relu(output)
        output = self.fc3(output)
        output = F.logsigmoid(output)
        return output


class RDPG:
    def __init__(self, n_features, n_a_hidden, n_c_hidden, a_lr, c_lr, memory_size, batch_size, gamma = 0.9, soft_replace = 0.1):
        self.n_features = n_features
        self.n_a_hidden = n_a_hidden
        self.n_c_hidden = n_c_hidden
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_replace = soft_replace

        self.a = ActorNet(n_features, n_a_hidden).to(device)
        self.q_sa = CriticNet(n_c_hidden).to(device)

        self.a_ = ActorNet(n_features, n_a_hidden).to(device)
        self.q_sa_ = CriticNet(n_c_hidden).to(device)

        # for param in self.a_.parameters():
        #     param.requires_grad = False

        # for param in self.q_sa_.parameters():
        #     param.requires_grad = False

        hard_update(self.a_, self.a)
        hard_update(self.q_sa_, self.q_sa)

        self.memory = deque(maxlen = self.memory_size)

        self.learn_step_counter = 0

        self.a_optimizer = torch.optim.Adam(self.a.parameters(), lr = self.a_lr)
        self.q_sa_optimizer = torch.optim.Adam(self.q_sa.parameters(), lr = self.c_lr)

    def choose_action(self, x):
        state = np.reshape(x, [-1, self.n_features])
        action = self.a(to_tensor(state, use_cuda=USE_CUDA))
        action = action.squeeze(1)
        action = to_numpy(action)
        return action

    def store_transition(self, state, action, reward, next_state):
        state, next_state = state[np.newaxis,:], next_state[np.newaxis,:]
        action,reward = np.array(action),np.array(reward)
        action = np.reshape(action, [1,-1])
        reward = np.reshape(reward, [1,-1])

        transition = np.concatenate((state, action, reward, next_state),axis = 1)
        self.memory.append(transition[0, :])

    def learn(self):
        if len(self.memory) == self.memory_size:

            if self.learn_step_counter % 200 ==0:
                soft_update(self.a_, self.a, self.soft_replace)
                soft_update(self.q_sa_, self.q_sa, self.soft_replace)

            self.learn_step_counter += 1

            self.a.zero_grad()
            self.q_sa.zero_grad()
            self.a_.zero_grad()
            self.q_sa_.zero_grad()

            batch = np.array(random.sample(self.memory, self.batch_size))
            batch_s = batch[:,:self.n_features]
            batch_a = batch[:,self.n_features:(self.n_features + 1)]
            batch_r = batch[:,(self.n_features + 1):(self.n_features + 2)]
            batch_s_ = batch[:,(self.n_features + 2):(self.n_features*2 + 2)]

            target_a = self.a_(to_tensor(batch_s_))
            next_q_value = self.q_sa_([to_tensor(batch_s_), target_a])
            next_q_value.volatile=False

            target_q = to_tensor(batch_r) + self.gamma * next_q_value

            current_q = self.q_sa([to_tensor(batch_s), to_tensor(batch_a)])

            value_loss = F.smooth_l1_loss(current_q, target_q)
            # value_loss /= self.batch_size


            action = self.a(to_tensor(batch_s))
            policy_loss = - self.q_sa([to_tensor(batch_s), action])
            # policy_loss /= self.batch_size

            value_loss.backward()
            

            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.q_sa_optimizer.step()
            self.a_optimizer.step()















