from envs import *
from modules import *
import torch

model_dir = './model/reference_clip__'

if __name__ == '__main__':
    env = Reference(render_mode='human')
    _, obs, _ = env.reset()
    model = MLPAgent(obs[0].reshape(-1).shape[0], env.n_actions[0])
    model.load_state_dict(torch.load(model_dir))
    
    for t in range(30):
        _, obs, _ = env.reset()
        truncation = termination = False
        total_reward = 0
        while (not truncation) and (not termination):
            feature = torch.as_tensor(obs, dtype=torch.float)
            action_values = model(feature)
            actions = torch.argmax(action_values, dim=1)
            actions = actions.cpu().numpy()

            state_, obs_, reward, termination, truncation, _ = env.step(actions)

            total_reward += reward
            state = state_
            obs = obs_
        print('Total Reward:%.4f' % (total_reward))