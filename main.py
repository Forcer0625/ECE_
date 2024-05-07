from qmix import QMIX_ECE
import torch
from envs import *

ep_steps = 25
total_steps = int(1e6)*ep_steps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.35, # more leads to slower decay
    'gamma':0.99,
    'lr': 1e-4,
    'tau':0.005, # more is harder
    'batch_size':256,
    'memory_size':1500000,
    'device':device,
    'episode_length':ep_steps,
    'logdir':'ece_reference',
}
ga_config = {
    'population_size':200,
    'crossover_rate':0.9,
    'mutation_rate':0.1,
}

if __name__ == '__main__':
    print(device)
    env = Reference()
    qmix = QMIX_ECE(env, config, ga_config)
    qmix.learn(total_steps)
    
    #qmix.save_model()


