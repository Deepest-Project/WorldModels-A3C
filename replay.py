""" Define controller """
import torch
import torch.nn as nn
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ExperienceReplay(object):
    def __init__(self, capacity=10000):
        super(ExperienceReplay, self).__init__()
        self.capacity = capacity
        self.memory = []
        self.idx = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = Transition(*args)
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_memory(self):
        return self.memory
    
    def reset(self):
        self.memory = []
        self.idx = 0
    
    def __len__(self):
        return len(self.memory)
    