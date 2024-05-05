import math
import torch
from utilis import ReplayBuffer
from envs import BaseMPE
from modules import MLPAgent
import random
from copy import deepcopy
from sga import *

class BaseRunner():
    def __init__(self, env:BaseMPE, policy:MLPAgent, replay_buffer:ReplayBuffer):
        self.env = env
        self.n_actions = env.n_actions[0]
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.device = policy.device

    def store(self, state, observation, action, reward, done, state_, observation_):
        self.replay_buffer.store(state, observation, action, reward, done, state_, observation_)

    def run(self):
        raise NotImplementedError

class EGreedyRunner(BaseRunner):
    def __init__(self, env, policy, replay_buffer, eps_start, eps_end, eps_dec):
        super().__init__(env, policy, replay_buffer)
        self.epsilon = self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec

    def update_epsilon(self, steps):
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * steps / self.eps_dec)
        
    def run(self, step):
        state, obs, _ = self.env.reset()
        truncation = termination = False
        total_reward = 0
        while (not truncation) and (not termination):
            actions = []
            for i in range(self.env.n_agents):
                if random.random() < self.epsilon:        
                    action = random.sample(range(self.n_actions), 1)[0]
                else:
                    feature = torch.as_tensor(obs[i], dtype=torch.float, device=self.device)
                    action_values = self.policy(feature)
                    action = torch.argmax(action_values)
                    action = action.item()
                actions.append(action)

            state_, obs_, reward, termination, truncation, _ = self.env.step(actions)
            total_reward += reward
            self.store(state, obs, actions, reward, termination, state_, obs_)

            state = state_
            obs = obs_

            step += 1
            self.update_epsilon(step)
            
        return total_reward, step

class ECERunner(EGreedyRunner):
    def __init__(self, env, policy, target_policy, mixer, target_mixer, replay_buffer, eps_start, eps_end, eps_dec, ga_config):
        super().__init__(env, policy, replay_buffer, eps_start, eps_end, eps_dec)
        self.target_policy = target_policy
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.ga = SGA(ga_config)

    def run(self, step):
        state, obs, _ = self.env.reset()
        truncation = termination = False
        total_reward = 0
        while (not truncation) and (not termination):
            if False:#random.random() < self.epsilon:
                actions, reward, termination, truncation, state_, obs_ = self.ga.run(self.env)
            else:
                feature = torch.as_tensor(obs, dtype=torch.float, device=self.device)
                action_values = self.policy(feature)
                actions = torch.argmax(action_values, dim=1)
                actions = actions.cpu().numpy()

                state_, obs_, reward, termination, truncation, _ = self.env.step(actions)

            self.store(state, obs, actions, reward, termination, state_, obs_)

            total_reward += reward
            state = state_
            obs = obs_

            step += 1
            self.update_epsilon(step)
            
        return total_reward, step
