import numpy as np

a = np.array(
    [
    [[0,0],[1,1],[2,2]],
    [[0,0],[1,1],[2,2]],
    [[3,3],[4,4],[5,5]],
    ]
)
b = a.reshape(-1, 2)
print(b)


exit()
from pettingzoo.mpe import simple_tag_v3, simple_spread_v3, simple_reference_v3
import time
env = simple_reference_v3.env()

while True:
    t = time.time()
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy
        env.step(action)
    env.close()
    print(time.time() - t)