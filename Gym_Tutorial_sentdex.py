import Box2D
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# https://www.youtube.com/watch?v=G-KvpNGudLw From this tutorial

LR = 1e-3

env = gym.make('CartPole-v1')
# env = gym.make('BipedalWalker-v2')

env.reset()
goal_steps = 500
score_requirement = 100
initial_games = 10000

# goal_steps = 500
# score_requirement = -10
# initial_games = 10000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

# some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # output = np.zeros(2)
                # output[data[1]] = 1
                training_data.append([data[0], data[1]])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data_save

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name="input")

    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 512, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 256, activation='relu')
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 128, activation='relu')
    # network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    # network = fully_connected(network, 4, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    # network = regression(network, optimizer='adam', learning_rate=LR, name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    Y = np.array([i[1] for i in training_data])

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=False, run_id='openaistuff')
    return model


def play_model(model, render=False):
    score = 0
    env.reset()
    prev_obs = []
    for i in range(goal_steps):
        if render:
            env.render()
        if len(prev_obs) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            # action = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            break
    return score

def get_training_population(model, score_requirement, games, mess_rate=0.2):
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(games):
        env.reset()
        score = 0
        game_memory = []
        prev_observation = []
        prev_obs = []
        for _ in range(goal_steps):
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
                # action = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]
            if random.random() < mess_rate:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = np.zeros(2)
                output[data[1]] = 1
                training_data.append([data[0], output])
        scores.append(score)

    training_data_save = np.array(training_data)
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data_save, min(accepted_scores)+1

def train(model, score_requirement, games, rounds):
    r = 0.1
    for _ in xrange(rounds):
        print("round {} of learning".format(_))
        training_data, score_requirement = get_training_population(model, score_requirement, games, r)
        r -= 0.01
        while training_data.shape[0] < 10000:
            training_data = np.repeat(training_data, 2, axis=0)
        model = train_model(training_data, model)
        play_model(model, True)
    return model


training_data = initial_population()
training_data = np.load(open('saved.npy'))
model = train_model(training_data)
model.save('Saved.model')
# X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
# model = neural_network_model(input_size = len(X[0]))
# model.load('Saved.model')

model = train(model, score_requirement, 1000, 1)
model.save('Saved.model')
play_model(model, True)


# scores = []
# choices = []
# for each_game in xrange(10):
#     score = 0
#     game_memory = []
#     prev_obs = []
#     env.reset()
#     for _ in range(goal_steps):
#         env.render()
#         if len(prev_obs) == 0:
#             action = random.randrange(0,2)
#         else:
#             # action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
#             action = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]
#         choices.append(action)
#         new_observation, reward, done, info = env.step(action)
#         prev_obs = new_observation
#         game_memory.append([new_observation, action])
#         score += reward
#         if done:
#             break
#     scores.append(score)
#
# print('Average Score', sum(scores)/len(scores))
# print('Choice1: {}, Choice 2: {}'.format(float(choices.count(1))/len(choices), float(choices.count(0))/len(choices)))
