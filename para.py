from pettingzoo.mpe import simple_spread_v3, simple_world_comm_v3
import time
from envs import Spread, Reference, Comm

env = simple_world_comm_v3.parallel_env(render_mode='human')#Comm('human')#
while True:
    t = time.time()
    observations, infos = env.reset()
    # truncations = terminations = False
    # while (not truncations) and (not terminations):
    #     actions = []
    #     for _ in range(3):
    #         actions.append(env.env.action_space('agent_0').sample())
    #     state, observations, rewards, terminations, truncations, infos = env.step(actions)
        #time.sleep(1.0)
        
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
    print(time.time() - t)
env.close()