#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:01:58 2019
@author: radwaelaraby
Modified on Sat May 9 21:37:00 2020
@author: has9h
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import struct
from textwrap import wrap
import time

NO_OF_VARIABLES = 1
NO_OF_ENCODING_CHARACHTERS = 32

OBJECTIVE_WEIGHT = 0.5
NO_OF_RUNS = 100
NO_OF_ITERATIONS = 100
NO_OF_AGENTS = 12
NO_OF_PARENTS = int(NO_OF_AGENTS / 3)
MUTATION_PROBABILITY = 0.1

MARKERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']


def run(run_i):
    """
    run
    """
    fitnesses, agents, agents_reduced = init()
    store_current_state(fitnesses, agents, agents_reduced)
    for i in range(NO_OF_ITERATIONS):
        fitnesses, agents, agents_reduced = genetic_algorithm(agents, fitnesses)
        if (i%20 == 0):
            store_current_state(fitnesses, agents, agents_reduced)
    #plot_summary('run_' + str(run_i))
    clear_state()
    return fittest_agent(fitnesses, agents, agents_reduced)


def init():
    """
    init
    """
    agents = np.random.uniform(-100.0, 100.0, (NO_OF_AGENTS, NO_OF_VARIABLES))
    fitnesses = compute_fitnesses(agents)
    agents_reduced = reduce_agents(agents)
    return [fitnesses, agents, agents_reduced]


def genetic_algorithm(agents, fitnesses):
    """
    genetic_algorithm
    """
    parents = select_parents(agents, fitnesses)
    children = breed(parents)
    new_agents = np.concatenate((parents, children))
    new_agents = mutate(new_agents)
    new_fitnesses = compute_fitnesses(new_agents)
    agents_reduced = reduce_agents(new_agents)
    return [new_fitnesses, new_agents, agents_reduced]


def select_parents(agents, fitnesses):
    """
    select_parents
    """
    parent_indexes = np.argpartition(fitnesses, NO_OF_PARENTS)[:NO_OF_PARENTS]
    return agents[parent_indexes]


def breed(parents):
    """
    breed
    """
    children = []
    for parent_i, parent_variables in enumerate(parents):
        mate_variables = select_mate_for(parents, parent_i)
        child_1, child_2 = crossover(parent_variables, mate_variables)
        children.append(child_1)
        children.append(child_2)
    return np.array(children)


def select_mate_for(all_candidates, parent_i):
    """
    select_mate_for
    """
    mate_candidates = np.concatenate((np.arange(0, parent_i), 
                                      np.arange(parent_i+1, NO_OF_PARENTS)))
    return all_candidates[random.choice(mate_candidates)]


def crossover(parent_1, parent_2):
    """
    crossover
    """
    crossover_point = select_crossover_point()
    parent_1_encoded = ''.join(np.vectorize(float_to_bin)(parent_1))
    parent_2_encoded = ''.join(np.vectorize(float_to_bin)(parent_2))
    child_1_encoded = ''.join((parent_1_encoded[:crossover_point],
                               parent_2_encoded[crossover_point:]))
    child_2_encoded = ''.join((parent_2_encoded[:crossover_point],
                               parent_2_encoded[crossover_point:]))
    child_1 = np.vectorize(bin_to_float)(wrap(child_1_encoded, NO_OF_ENCODING_CHARACHTERS))
    child_2 = np.vectorize(bin_to_float)(wrap(child_2_encoded, NO_OF_ENCODING_CHARACHTERS))
    return [child_1, child_2]


def select_crossover_point():
    """
    select_crossover_point
    """
    length = NO_OF_VARIABLES * NO_OF_ENCODING_CHARACHTERS
    return random.randrange(1, length)


def compute_fitnesses(agents):
    """
    compute_fitnesses
    """
    fitnesses = []
    for a_i, agent_variables in enumerate(agents):
        f = (OBJECTIVE_WEIGHT * np.power(agent_variables[0], 2)) +\
            ((1 - OBJECTIVE_WEIGHT) * np.power(agent_variables[0] - 2, 2))
        fitnesses.append(f)
    return np.array(fitnesses)


def mutate(agents):
    """
    mutate
    """
    for a_i in range(NO_OF_AGENTS):
        for v_i in range(NO_OF_VARIABLES):
            if (random.random() < MUTATION_PROBABILITY):
                agents[a_i][v_i] += random.uniform(-10, 10)
    return agents


def fittest_agent(fitnesses, agents, agents_reduced):
    """
    fittest_agent
    """
    a_i = np.argmin(fitnesses)
    if (NO_OF_VARIABLES == 1):
        reduced = [agents_reduced[0][a_i]]
    else:
        reduced = [agents_reduced[0][a_i], agents_reduced[1][a_i]]
    return [
            fitnesses[a_i],
            agents[a_i],
            reduced
        ]


def reduce_agents(agents):
    """
    reduce_agents
    """
    if (NO_OF_VARIABLES == 1):
        return agents.transpose()
    pca = PCA(n_components=2)
    return pca.fit_transform(agents).transpose()


def store_current_state(fitnesses, agents, agents_reduced):
    """
    store_current_state
    """
    fit_fitness, fit_agent, fit_reduced = fittest_agent(fitnesses, agents, agents_reduced)
    accumulate_fitness(fit_fitness)
    accumulate_agents_reduced(agents_reduced)    


def accumulate_fitness(fitness):
    """
    accumulate_fitness
    """
    if(hasattr(accumulate_fitness, 'data')==False):
        accumulate_fitness.data = []
    accumulate_fitness.data.append(fitness)


def accumulate_agents_reduced(agents_reduced):
    """
    accumulate_agents_reduced
    """
    if(hasattr(accumulate_agents_reduced, 'data')==False):
        accumulate_agents_reduced.data = []
    accumulate_agents_reduced.data.append(agents_reduced)


def clear_state():
    """
    clear_state
    """
    accumulate_fitness.data = []
    accumulate_agents_reduced.data = []


def plot_summary(fig_i):
    """
    plot_summary
    """
    plt.figure(fig_i + '_' + str(time.time()))
    plt.subplot(1, 2, 1)
    plt.plot(accumulate_fitness.data)
    plt.subplot(1, 2, 2)
    for a_i, data in enumerate(accumulate_agents_reduced.data):
        if (NO_OF_VARIABLES == 1):
            plt.scatter(range(NO_OF_AGENTS), data[0], marker=getMarker(a_i))
        else:
            plt.scatter(data[0], data[1], marker=getMarker(a_i))
    plt.show()


def getMarker(i):
    """
    getMarker
    """
    return "$"+MARKERS[i % len(MARKERS)]+"$"


def float_to_bin(num):
    """
    float_to_bin
    """
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)


def bin_to_float(binary):
    """
    bin_to_float
    """
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


"""
Execution
"""
solutions = []
figure_name = 'Overview_' + str(time.time())
for r_i in range(NO_OF_RUNS):
    OBJECTIVE_WEIGHT = r_i / 100
    b_fitness, b_agent, b_agent_reduced = run(r_i)
    solutions.append(b_agent)
    plt.figure(figure_name)
    if (NO_OF_VARIABLES == 1):
        plt.scatter(r_i, b_agent_reduced[0])
    else:
        plt.scatter(b_agent_reduced[0], b_agent_reduced[1])
    print('#', r_i, b_fitness, b_agent, b_agent_reduced)
plt.show()
print('min:', min(solutions))
print('max:', max(solutions))
