import numpy as np
import random

class Individual():
    def __init__(self):
        self.fitness = None

    def random_actions(self, n_agents, n_actions, n_episode_steps):
        self.actions = random.choices(population=list(range(-1, n_actions)), k=n_agents*n_episode_steps)
        self.actions = np.array(self.actions)

    def set_actions(self, actions=None, n_agents=None, n_actions=None, n_episode_steps=None):
        if actions is None:
            self.random_actions(n_agents, n_actions, n_episode_steps)
        else:
            self.actions = actions

    def __str__(self):
        return self.actions
    
    def __len__(self):
        return len(self.actions)
    
class SGA():
    def parent_select(population:list, k=2):
        n_population = len(population)
        all_idx = list(range(n_population))
        n_parent_pair = n_population//2
        parent_idx = []

        for _ in range(n_parent_pair):
            comps = random.sample(all_idx, k=2*k)
            parent1 = comps[:k].sort(key=lambda i: population[i].fitness, reverse=True)[0]
            parent2 = comps[k:].sort(key=lambda i: population[i].fitness, reverse=True)[0]

            parent_idx.append([parent1, parent2])
        
        return parent_idx
        
    def _crossover(parent1:Individual, parent2:Individual, crossover_rate=0.9):
        a = parent1.actions
        b = parent2.actions

        if random.random() <= crossover_rate:
            cut_index = random.choices(list(range(len(parent1))), k=2)
            
            child_a = np.concatenate(a[:cut_index[0]] + b[cut_index[0]:cut_index[1]] + a[cut_index[1]:], axis=0)
            child_b = np.concatenate(b[:cut_index[0]] + a[cut_index[0]:cut_index[1]] + b[cut_index[1]:], axis=0)

        else:
            child_a = a.copy()
            child_b = b.copy()

        offspring_a = Individual(random_init=False)
        offspring_b = Individual(random_init=False)
        offspring_a.set_actions(child_a)
        offspring_b.set_actions(child_b)
        
        return offspring_a, offspring_b
            
    def crossover(population:list, parent_idx, crossover_rate=0.9):
        for parent1, parent2 in parent_idx:
            SGA._crossover(population[parent1], population[parent2], crossover_rate)
    
    def _mutate(offspring:Individual, mutation_rate=0.1, low=None, high=None):
        for i in range(len(offspring)):
            if random.random() <= mutation_rate:
                offspring.actions[i] = random.choice(range(low, high), k=1)[0]

    def mutation(offsprings:list, mutation_rate=0.1, low=None, high=None):
        for individual in offsprings:
            SGA._mutate(individual, mutation_rate, low, high)