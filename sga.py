import numpy as np
import random

class Individual():
    def __init__(self, n_agents, n_actions, n_episode_steps):
        self.fitness = None
        self.actions = random.choices(population=list(range(-1, n_actions)), k=n_agents*n_episode_steps)
        self.actions = np.array(self.actions)

    def __str__(self):
        return self.actions
    
    def __len__(self):
        return len(self.actions)
    
class SGA():
    pass