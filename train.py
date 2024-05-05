from qmix import QMIX
import torch
from envs import *
from copy import deepcopy

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
    'logdir':'reference_clip',
}

if __name__ == '__main__':
    print(device)
    for t in range(30):
        new_config = deepcopy(config)
        new_config['logdir'] = config['logdir']+str(t)
        env = Reference()
        qmix = QMIX(env, new_config)
        qmix.learn(total_steps)
    
    #qmix.save_model()


