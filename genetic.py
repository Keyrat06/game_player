import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter
import time
import matplotlib.pyplot as plt
plt.ion()
import copy

def some_random_games_first(GOAL_STEPS, env):
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
    def __init__(self, input_size, hiddens, output_size, activation = relu):
        self.i = 0
        self.Omega1 = random.random() * np.pi * 0.5
        self.Omega2 = random.random() * np.pi * 0.5
        self.activation = activation
        self.input_size = input_size
        self.hiddens = hiddens
        self.output_size = output_size
        self.num_layers = len(hiddens) + 1
        self.weights = dict()
        self.weights[0] = np.random.standard_normal((input_size+4, hiddens[0]))/input_size**0.5
        for layer in xrange(1, self.num_layers-1):
            self.weights[layer] = np.random.standard_normal((hiddens[layer-1]+1, hiddens[layer]))/hiddens[layer-1]**0.5
        self.weights[self.num_layers-1] = np.random.standard_normal((hiddens[-1]+1, output_size))/hiddens[-1]**0.5

    def predict(self, input):
        self.i += 1
        hidden = self.activation(np.matmul(np.hstack((input, 1*np.sin(self.i * self.Omega1), 1*np.sin(self.i * self.Omega2), self.i*0.01, 1)), self.weights[0]))
        for layer in xrange(1, self.num_layers-1):
            hidden = self.activation(np.matmul(np.hstack((hidden,  1)), self.weights[layer]))
        last_layer = np.matmul(np.hstack((hidden, 1)), self.weights[self.num_layers-1])
        return last_layer

def mix(Net_1, Net_2, mix_params=[0.5,0.5], noise = 0.01, net_type = Genetic_Net_discrete):
    baby_Net = net_type(Net_1.input_size, Net_1.hiddens, Net_1.output_size)
    for layer in Net_1.weights:
        for column in xrange(len(baby_Net.weights[layer])):
            baby_Net.weights[layer][column] = copy.deepcopy(np.random.choice((Net_1, Net_2), 1, p=mix_params)[0].weights[layer][column])
        baby_Net.weights[layer] += np.random.standard_normal(baby_Net.weights[layer].shape)/baby_Net.weights[layer].shape[0]**0.5 * noise
    if net_type == Genetic_Net_discrete:
        return baby_Net
    else:
        baby_Net.Omega1 = np.random.choice((Net_1, Net_2), 1, p=mix_params)[0].Omega1 + random.gauss(0,1) * noise * np.pi
        baby_Net.Omega2 = np.random.choice((Net_1, Net_2), 1, p=mix_params)[0].Omega2 + random.gauss(0,1) * noise * np.pi
        return baby_Net

def copy_net_with_noise(Net, net_type = Genetic_Net_discrete):
    baby_Net = net_type(Net.input_size, Net.hiddens, Net.output_size)
    for layer in Net.weights:
        for column in xrange(len(baby_Net.weights[layer])):
            baby_Net.weights[layer][column] = copy.deepcopy(Net.weights[layer][column])
        baby_Net.weights[layer] += np.random.standard_normal(baby_Net.weights[layer].shape)/baby_Net.weights[layer].shape[0]**0.5 * 0.01 * (100 if random.random() < 0.05 else 1)
    if net_type == Genetic_Net_discrete:
        return baby_Net
    else:
        baby_Net.Omega1 = Net.Omega1 + random.gauss(0,1) * 0.01 * np.pi * (100 if random.random() < 0.05 else 1)
        baby_Net.Omega2 = Net.Omega2 + random.gauss(0,1) * 0.01 * np.pi * (100 if random.random() < 0.05 else 1)
        return baby_Net

def get_initial_population(num, input_size, hiddens, output_size, net_type = Genetic_Net_discrete):
    population = []
    for _ in xrange(num):
        population.append(net_type(input_size, hiddens, output_size))
    return np.array(population)


def play_game(model, env, render = False, steps=100):
    score = 0
    env.reset()
    prev_obs = []
    for i in range(steps):
        if render:
            env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = model.predict(prev_obs)
        new_observation, reward, done, info = env.step(action)
        if type(new_observation) is int:
            one_hot = ([0.0] * model.input_size)
            one_hot[new_observation] = 1.0
            new_observation = one_hot
        prev_obs = new_observation
        score += reward #+ 0.01/(i+1) * random.random()
        if done:
            break
    return min((200.0, score))


def test_population(population, population_size, env, steps):
    scores = []
    for i in xrange(population_size):
        scores.append(play_game(population[i], env, False, steps))
    return np.array(scores)

def reproduce(survivors, positive_scores, population_size, net_type = Genetic_Net_discrete):
    population = list(survivors)
    total_positive_scores = float(positive_scores.sum())
    while len(population) < population_size:
        if random.random() < 0.3:
            net_1_l, net_2_l = np.random.choice(range(len(survivors)), 2, replace=False, p=(positive_scores/total_positive_scores))
            net_1, net_2 = survivors[net_1_l], survivors[net_2_l]
            scores = np.array([positive_scores[net_1_l], positive_scores[net_2_l]])
            population.append(mix(net_1, net_2, mix_params=scores/scores.sum(), net_type = net_type))
        if random.random() < 0.9:
            net_1_l, net_2_l = np.random.choice(range(len(survivors)), 2, replace=False, p=(positive_scores/total_positive_scores))
            net_1, net_2 = survivors[net_1_l], survivors[net_2_l]
            population.append(copy_net_with_noise(net_1, net_type))
            population.append(copy_net_with_noise(net_2, net_type))
        else:
            model_net = survivors[0]
            population.append(net_type(model_net.input_size, model_net.hiddens, model_net.output_size))
    return np.array(population)

def train(population, generations, keep_porportion, population_size, env, net_type = Genetic_Net_discrete, game_steps=100):
    avg_scores = [0]
    max_scores = [0]
    for i in xrange(generations):
        scores = test_population(population, population_size, env, game_steps)
        print("Average score in generation {} is {}, max score is {}".format(i, scores.sum()/len(scores), scores.max()))
        avg_scores.append(scores.sum()/len(scores)), max_scores.append(scores.max())
        positive_scores = (scores - scores.min()) + random.random()*1e-6
        # print positive_scores
        # survivors_inx = np.random.choice(population_size, int(population_size*keep_porportion), replace=True, p=(positive_scores/float(positive_scores.sum())))
        survivors_inx = sorted(range(population_size), key=lambda x: -positive_scores[x])[0:int(population_size*keep_porportion)]
        survivors = population[survivors_inx]
        survivors_scores = np.array(positive_scores)[survivors_inx]
        viewer = population[np.argmax(scores)]
        population = reproduce(survivors, survivors_scores, population_size, net_type=net_type)
        plt.cla()
        plt.plot(avg_scores, label="Average Population Scores")
        plt.plot(max_scores, label="Best Population Score")
        plt.title("Genetic Algorithm Scores on FrozenLake enviroment")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.legend()
        plt.pause(0.0001)
        play_game(viewer, env, True, game_steps)
    return avg_scores, max_scores




