import Box2D
import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter
import time

POPULATION_SIZE = 100
KEEP_PORPORTION = 0.25

#for BipedalWalker-v2
INPUT_SIZE = 28
OUTPUT_SIZE = 4

#
# INPUT_SIZE = 4
# OUTPUT_SIZE = 2

HIDDENS = [10, 10]
GOAL_STEPS = 500
GENERATIONS = 100

env = gym.make('BipedalWalker-v2')
# env = gym.make('CartPole-v1')
env.reset()

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(GOAL_STEPS):
            env.render()
            action = env.action_space.sample()
            print action
            print "HERE"
            observation, reward, done, info = env.step(action)
            print "observation:", observation
            print "reward:", reward
            print "info", info
            print "done", done
            if done:
                break

# some_random_games_first()

def relu(x):
    return np.maximum(x, 0)

def linear(x):
    return x

class Genetic_Net_discrete(object):
    def __init__(self, input_size, hiddens, output_size, activation = linear):
        self.activation = activation
        self.input_size = input_size
        self.hiddens = hiddens
        self.output_size = output_size
        self.num_layers = len(hiddens) + 1
        self.weights = dict()
        self.weights[0] = np.random.standard_normal((input_size+1, hiddens[0]))/input_size**0.5
        for layer in xrange(1, self.num_layers-1):
            self.weights[layer] = np.random.standard_normal((hiddens[layer-1]+1, hiddens[layer]))/hiddens[layer-1]**0.5
        self.weights[self.num_layers-1] = np.random.standard_normal((hiddens[-1]+1, output_size))/hiddens[-1]**0.5

    def predict(self, input):
        hidden = self.activation(np.matmul(np.hstack((input, 1)), self.weights[0]))
        for layer in xrange(1, self.num_layers-1):
            hidden = self.activation(np.matmul(np.hstack((hidden, 1)), self.weights[layer]))
        last_layer = np.matmul(np.hstack((hidden, 1)), self.weights[self.num_layers-1])
        return np.argmax(last_layer)

class Genetic_Net_continous(object):
    def __init__(self, input_size, hiddens, output_size, activation = linear):
        self.i = 0
        self.Omega1 = random.random() * np.pi
        self.Omega2 = random.random() * np.pi
        self.activation = activation
        self.input_size = input_size
        self.hiddens = hiddens
        self.output_size = output_size
        self.num_layers = len(hiddens) + 1
        self.weights = dict()
        self.weights[0] = np.random.standard_normal((input_size+3, hiddens[0]))/input_size**0.5
        for layer in xrange(1, self.num_layers-1):
            self.weights[layer] = np.random.standard_normal((hiddens[layer-1]+1, hiddens[layer]))/hiddens[layer-1]**0.5
        self.weights[self.num_layers-1] = np.random.standard_normal((hiddens[-1]+1, output_size))/hiddens[-1]**0.5

    def predict(self, input):
        self.i += 1
        hidden = self.activation(np.matmul(np.hstack((input, np.sin(self.i * self.Omega1), np.sin(self.i * self.Omega2), 1)), self.weights[0]))
        for layer in xrange(1, self.num_layers-1):
            hidden = self.activation(np.matmul(np.hstack((hidden,  1)), self.weights[layer]))
        last_layer = np.matmul(np.hstack((hidden, 1)), self.weights[self.num_layers-1])
        return last_layer

def mix(Net_1, Net_2, mix_params=[0.5,0.5], noise = 0.01, net_type = Genetic_Net_discrete):
    baby_Net = net_type(Net_1.input_size, Net_1.hiddens, Net_1.output_size)
    for layer in Net_1.weights:
        for column in xrange(len(baby_Net.weights[layer])):
            baby_Net.weights[layer][column] = np.random.choice((Net_1, Net_2), 1, p=mix_params)[0].weights[layer][column]
        baby_Net.weights[layer] += np.random.standard_normal(baby_Net.weights[layer].shape)/baby_Net.weights[layer].shape[0]**0.5 * noise
    baby_Net.Omega1 = (Net_1.Omega1 + Net_2.Omega1)/2 + np.random.standard_normal(1)[0]
    baby_Net.Omega2 = (Net_1.Omega2 + Net_2.Omega2)/2 + np.random.standard_normal(1)[0]
    return baby_Net

def get_initial_population(num, input_size, hiddens, output_size, net_type = Genetic_Net_discrete):
    population = []
    for _ in xrange(num):
        population.append(net_type(input_size, hiddens, output_size))
    return np.array(population)

def play_game(model, render = False, steps=GOAL_STEPS):
    score = 0
    env.reset()
    prev_obs = []
    for i in range(steps):
        if render:
            env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = model.predict(np.array(list(prev_obs) + [3*random.random(), 3*random.random(), 10*np.sin(i*0.2*np.pi), 10*np.sin(i*0.05*np.pi)]))
            # action = model.predict(prev_obs) #used this for cart_and_stick
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            break
    return score

def test_population(population):
    scores = []
    for i in xrange(POPULATION_SIZE):
        scores.append(play_game(population[i], False))
    return np.array(scores)

def reproduce(survivors, scores, net_type = Genetic_Net_discrete):
    population = list(survivors)
    positive_scores = scores
    if scores.min() < 0:
        positive_scores = scores - scores.min()
    total_positiv_scores = float(positive_scores.sum())
    for _ in xrange(POPULATION_SIZE-len(survivors)):
        net_1, net_2 = np.random.choice(survivors, 2, replace=False, p=(positive_scores/total_positiv_scores))
        population.append(mix(net_1, net_2, net_type = net_type))
    return np.array(population)

def train(population, net_type = Genetic_Net_discrete):
    for i in xrange(GENERATIONS):
        scores = test_population(population)
        print("Best score in generation {} is {}".format(i, scores.max()))
        positive_scores = scores-scores.min()
        survivors_inx = np.random.choice(POPULATION_SIZE, int(POPULATION_SIZE*KEEP_PORPORTION), replace=False, p=(positive_scores/float(positive_scores.sum())))
        survivors = population[survivors_inx]
        survivors_scores = np.array(scores)[survivors_inx]
        population = reproduce(survivors, survivors_scores, net_type=net_type)
        play_game(population[np.argmax(scores)], True, steps=1000)

##### For Bipedal Walk
population = get_initial_population(POPULATION_SIZE, INPUT_SIZE, HIDDENS, OUTPUT_SIZE, net_type=Genetic_Net_continous)
population = train(population, net_type=Genetic_Net_continous)

##### For Cart Pole
# population = get_initial_population(POPULATION_SIZE, INPUT_SIZE, HIDDENS, OUTPUT_SIZE, net_type=Genetic_Net_discrete)
# population = train(population, net_type=Genetic_Net_discrete)


