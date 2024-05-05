import numpy as np
from collections import deque
from copy import deepcopy

class FrameStack():
    def __init__(self, n_stacks:int=4):
        self.n_stacks = n_stacks
        self.frame_buffer = deque(maxlen=n_stacks)

    def get(self):
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames.reshape(-1)

    def push(self, image:np.ndarray):
        self.frame_buffer.append(image)
        while len(self.frame_buffer) < self.n_stacks:
            self.frame_buffer.append(image)

    def render(self):
        pass

    def clear(self):
        self.frame_buffer.clear()
    
    def next_frame(self, image:np.ndarray):
        '''Return stacked frames the next frame'''
        temp = deepcopy(self.frame_buffer)
        temp.append(image)
        return np.stack(temp, axis=0).reshape(-1)

class MultiFrameStack():
    def __init__(self, agents, n_stacks:int=4):
        self.frame_buffers = {}
        self.agents = agents
        for agent in self.agents:
            self.frame_buffers[agent] = FrameStack(n_stacks)

    def get(self):
        ''' ndarray: [agent, observation]'''
        observations = []
        for agent in self.agents:
            observations.append(self.frame_buffers[agent].get())
        return np.array(observations)

    def push(self, state):
        for agent in self.agents:
            self.frame_buffers[agent].push(state[agent])

    def top(self):
        states = []
        for agent in self.agents:
            states.append(self.frame_buffers[agent].frame_buffer[-1])
        return np.array(states)

    def clear(self):
        for agent in self.agents:
            self.frame_buffers[agent].clear()
    
    def next_frame(self, single_frame):
        state = []
        observations = []
        for agent in self.agents:
            observations.append(self.frame_buffers[agent].next_frame(single_frame[agent]))
            state.append(single_frame[agent])
        return np.array(state), np.array(observations)
        
    
class ReplayBuffer():
    def __init__(self, buffer_size:int, state_shape, observation_shape, n_agents):
        self.max_buffer_size = buffer_size
        self.index = 0
        self.buffer_size = 0
        self.states = np.zeros((self.max_buffer_size, *state_shape))
        self.states_= np.zeros((self.max_buffer_size, *state_shape))
        self.observations = np.zeros((self.max_buffer_size, n_agents, *observation_shape))
        self.observations_= np.zeros((self.max_buffer_size, n_agents, *observation_shape))
        self.actions= np.zeros((self.max_buffer_size, n_agents), dtype=int)
        self.rewards= np.zeros(self.max_buffer_size)
        self.dones  = np.zeros(self.max_buffer_size, dtype=bool)

    def store(self, state, observation, action, reward, done, state_, observation_):
        self.states [self.index] = state
        self.states_[self.index] = state_
        self.observations [self.index] = observation
        self.observations_[self.index] = observation_
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones  [self.index] = done

        self.index = (self.index+1)%self.max_buffer_size
        self.buffer_size = min(self.buffer_size+1, self.max_buffer_size)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer_size, batch_size)

        states = self.states [batch]
        states_= self.states_[batch]
        observations = self.observations [batch]
        observations_= self.observations_[batch] 
        actions= self.actions[batch]
        rewards= self.rewards[batch]
        dones  = self.dones  [batch]

        return states, observations, actions, rewards, dones, states_, observations_
    
    def __len__(self):
        return self.buffer_size