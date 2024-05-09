import torch
import torch.nn as nn
import numpy as np
from modules import QMixer, MLPAgent
from utilis import ReplayBuffer
from runner import *
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
        observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device).view(-1, *observations_[0][0].shape)

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
    def __init__(self, env:BaseMPE, config, ga_config):
        super().__init__(env, config)
        self.runner = ECERunner(self.env, self.policy, self.target_policy, self.mixer, self.target_mixer,\
                                self.memory, self.config['eps_start'], self.config['eps_end'], self.config['eps_dec'], ga_config)
        
class QMIX_ECE_v2(QMIX):
    def __init__(self, env: BaseMPE, config, ga_config):
        super().__init__(env, config)
        self.n_episode_length = config['episode_length']
        self.ga_config = ga_config

    def learn(self, total_steps):
        step = generation = 0
        populations = []
        n_populations = self.ga_config['population_size']
        for _ in range(n_populations):
            individual = Individual()
            individual.set_actions(None, self.n_agents, self.n_actions, self.n_episode_length)
            populations.append(individual)
        
        while step < total_steps:
            infos = []
            self.seeding()
            # evaluate
            with torch.no_grad():
                for individual in populations:
                    fitness, episode_steps, mean_td_error, total_reward = self.evaluate(individual.actions)
                    individual.fitness = fitness
                    #step += episode_steps

                    info = {
                        'Fitness':fitness,
                        'TD-Error':mean_td_error,
                        'Ep.Reward':total_reward,
                        'Epislon':self.runner.epsilon,
                    }
                    infos.append(info)
            step += episode_steps*n_populations
            
            # RL side
            for i in range(n_populations):
                loss = self.update()
                self.sync()
                infos[i]['Loss'] = loss
                self.log_info(generation*n_populations + i, infos[i])

            # GA side
            populations.sort(key=lambda individual: individual.fitness, reverse=True)
            populations = populations[:n_populations]
            
            parents_idx = SGA.parent_select(populations, k=10)

            offsprings = SGA.crossover(populations, parents_idx, self.ga_config['crossover_rate'])
            
            SGA.mutation(populations, self.ga_config['mutation_rate'], low=-1, high=self.n_actions)
            
            populations = populations + offsprings # (mu+lambda)

            generation += 1

            print('Steps: %d\tEpsilon:%.2f\tEp.Reward: %.2f\tFitness: %.2f\tTD-Error: %.2f' % (step, self.runner.epsilon, total_reward, info['Fitness'], info['TD-Error']))

        torch.save(self.infos, './log/'+self.config['logdir'])
        
    def seeding(self):
        self.seed = random.randint(0, 2**31-1)

    def caculate_td_error(self, episode_idx):
        states, observations, actions, rewards,\
            dones, states_, observations_ = self.memory.last_episode(episode_idx)

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
        loss = self.loss(q_tot, target)

        return loss.item()

    def evaluate(self, all_actions):
        state, obs, _ = self.env.reset(seed=self.seed)
        truncation = termination = False
        total_reward = step = 0
        n_agents = self.env.n_agents
        episode_idx = []

        while (not truncation) and (not termination):
            actions = all_actions[step*n_agents: step*n_agents+n_agents]
            encoded_actions = []
            for i in range(self.n_agents):
                if actions[i] == -1:
                    feature = torch.as_tensor(obs[i], dtype=torch.float, device=self.policy.device)
                    action_values = self.policy(feature)
                    a = torch.argmax(action_values).cpu().item()
                else:
                    a = actions[i]
                encoded_actions.append(a)
            
            state_, obs_, reward, termination, truncation, _ = self.env.step(encoded_actions)

            index = self.memory.store(state, obs, encoded_actions, reward, termination, state_, obs_)
            episode_idx.append(index)

            total_reward += reward

            state = state_
            obs = obs_

            step += 1

        mean_td_error = self.caculate_td_error(episode_idx)
        epsilon = self.runner.epsilon
        return epsilon*mean_td_error*step + (1-epsilon)*total_reward, step, mean_td_error, total_reward