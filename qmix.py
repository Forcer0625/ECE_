import torch
import torch.nn as nn
import numpy as np
from modules import QMixer, MLPAgent
from utilis import ReplayBuffer
from runner import EGreedyRunner, ECERunner
from envs import BaseMPE
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

class QMIX():
    def __init__(self, env:BaseMPE, config):
        self.env = env
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions[0]
        state, observation, _ = self.env.reset()

        self.batch_size = config['batch_size']
        self.memory_size = config['memory_size']
        self.memory = ReplayBuffer(self.memory_size, state.shape, observation[0].shape, self.n_agents)

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.device = config['device']
        self.loss = nn.MSELoss()
        
        self.policy = MLPAgent(observation[0].reshape(-1).shape[0], self.n_actions, device=self.device)
        self.target_policy = deepcopy(self.policy)

        self.mixer = QMixer(state.shape[0], self.n_agents, device=self.device)
        self.target_mixer = deepcopy(self.mixer)
        
        self.parameters = list(self.policy.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)

        self.runner = EGreedyRunner(env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])
        

        self.logger = SummaryWriter('./runs/'+config['logdir'])
        self.infos = []
        self.config = config
    
    def log_info(self, step, info:dict):
        for key in info.keys():
            self.logger.add_scalar('Train/'+key, info[key], step)
        self.infos.append(info)
        #self.logger.add_scalar('Train/Epsilon', self.runner.epsilon, step)
        #self.logger.add_scalar('Train/Loss', loss, step)
        

    def learn(self, total_steps):
        x = 0
        step = 0
        mean_reward = []
        
        while step < total_steps:
            with torch.no_grad():
                total_reward, step = self.runner.run(step)
            mean_reward.append(total_reward)

            if len(self.memory) < self.batch_size:
                continue
            
            loss = self.update()

            self.sync()
            
            info = {
                'Ep.Reward':total_reward,
                'Epsilon':self.runner.epsilon,
                'Loss':loss,
            }
            self.log_info(x, info)
            
            x+=1
            if x % 1000 == 0:
                print('Steps: %d\tEpsilon:%.2f\tEp.Reward: %.2f\tAve.Reward: %.2f' % (step, self.runner.epsilon, total_reward, np.mean(mean_reward[-100:])))
        torch.save(self.infos, './log/'+self.config['logdir'])

    def update(self):
        states, observations, actions, rewards,\
            dones, states_, observations_ = self.memory.sample(self.batch_size)
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device).view(-1, *observations[0][0].shape)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).view(-1, self.n_agents)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device)
        states_= torch.as_tensor(states_, dtype=torch.float32, device=self.device)
        observations_= torch.as_tensor(observations_, dtype=torch.float, device=self.device).view(-1, *observations_[0][0].shape)

        action_values = self.policy(observations).reshape(-1, self.n_agents, self.n_actions)
        action_values = action_values.gather(2, actions.unsqueeze(2))
        action_values = action_values.reshape(-1, 1, self.n_agents)

        # double-q
        with torch.no_grad():
            estimate_action_values = self.policy(observations_).reshape(-1, self.n_agents, self.n_actions)
            next_action = torch.max(estimate_action_values, dim=2).indices
            next_action_values = self.target_policy(observations_).reshape(-1, self.n_agents, self.n_actions)
            next_action_values = next_action_values.gather(2, next_action.unsqueeze(2))
            next_action_values = next_action_values.reshape(-1, 1, self.n_agents)

        #mixer
        q_tot = self.mixer(action_values, states).squeeze()
        target_q_tot = self.target_mixer(next_action_values, states_).squeeze()

        # calculate loss
        target = rewards + self.gamma * (1 - dones) * target_q_tot
        loss = self.loss(q_tot, target.detach())

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, 10)
        self.optimizer.step()

        return loss.item()

    def sync(self):
        # update agent
        target_net_weights = self.target_policy.state_dict()
        q_net_weights = self.policy.state_dict()
        for key in q_net_weights:
            target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
        self.target_policy.load_state_dict(target_net_weights)

        # update mixer
        target_net_weights = self.target_mixer.state_dict()
        q_net_weights = self.mixer.state_dict()
        for key in q_net_weights:
            target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
        self.target_mixer.load_state_dict(target_net_weights)
    
    def save_model(self, path=None):
        if path is None:
            path = './model/'+self.config['logdir']
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path=None):
        if path is None:
            path = './model/'+self.config['logdir']
        self.policy.load_state_dict(torch.load(path))

class QMIX_ECE(QMIX):
    def __init__(self, env:BaseMPE, config):
        super.__init__(env, config)
        self.runner = ECERunner(self.env, self.policy, self.memory)

    