import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, n_agents, state_size, action_size, buffer_size, batch_size):
        """
        Fixed-size buffer to store experience tuples.
        :param state_size: (int)
        :param action_size: (int)
        :param buffer_size: (int)
        :param batch_size: (int)
        """
        self.memory = {
            'state': np.zeros((n_agents, buffer_size, state_size), dtype=np.float32),
            'action': np.zeros((n_agents, buffer_size, action_size), dtype=np.float32),
            'reward': np.zeros((n_agents, buffer_size), dtype=np.float32),
            'next_state': np.zeros((n_agents, buffer_size, state_size), dtype=np.float32),
            'done': np.zeros((n_agents, buffer_size), dtype=np.float32)
        }
        self.memory_keys = set(self.memory.keys())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def add(self, experience_dict):
        """
        Add a new experience to memory.
        :param experience_dict: experience dictionary with keys {state, action, reward, next_state, done}
        """
        assert self.memory_keys == set(experience_dict.keys())
        for k in self.memory_keys:
            self.memory[k][:, self.ptr] = experience_dict[k]
        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def reset(self):
        self.ptr = 0
        self.size = 0

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        idx = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=False)
        out = {k: torch.from_numpy(self.memory[k][:, idx]) for k in self.memory_keys}
        return out

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return self.size
