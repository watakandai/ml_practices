import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = device
    
    def append(self, state, action, n_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward 
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)
    
    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )