from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_world_comm_v3
from utilis import MultiFrameStack


class BaseMPE():
    def __init__(self, render_mode=None):
        self.env = self.n_agents = self.n_actions = None
        self.frame_buffer = None
    
    def step(self, actions):
        '''returns global_state, observations, reward, terminations, truncations, infos'''
        dict_actions = {}
        a = 0
        for agent in self.env.agents:
            dict_actions[agent] = actions[a]
            a+=1
        single_frame_observatons, reward, terminations, truncations, infos = self.env.step(dict_actions)
        self.frame_buffer.push(single_frame_observatons)
        terminations = terminations['agent_0']
        truncations = truncations['agent_0']
        reward = reward['agent_0']
        return self.global_state(), self.frame_buffer.get(), reward, terminations, truncations, infos

    def reset(self):
        self.frame_buffer.clear()
        single_frame_observatons, infos = self.env.reset()
        self.frame_buffer.push(single_frame_observatons)
        observatons = self.frame_buffer.get()
        return self.global_state(), observatons, infos

    def global_state(self):
        '''global state is concated from all agents' local observations'''
        return self.frame_buffer.top().reshape(-1)
    
class Spread(BaseMPE):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_spread_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.n_actions = [self.env.action_space(agent).n for agent in self.env.agents]
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

class Reference(BaseMPE):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_reference_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.n_actions = [self.env.action_space(agent).n for agent in self.env.agents]
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

class Comm(BaseMPE):
    '''inhomogeneous agents : [leadadversary_0, adversary_0, adversary_1, adversary_3, agent_0, agent_1]'''
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_world_comm_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

    def step():
        pass
    def reset():
        pass
    def global_state():
        pass

        