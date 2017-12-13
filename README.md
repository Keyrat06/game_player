# game_player
Raoul Khouri, Ryan Kelly, and Micheal DeLaus 6.867 final project code repository

For all this code to run you need the following python modules

numpy
matplotlib
Tensorflow
tflearn
gym
box2d

1. Gym_tutorial_sentdex.py
  * A random intialization and optimization implementation for CartPole enviroment
1. Reinforcement_ice_walker.py
  * Action-Critic Reinforcement model found to work on BipedalWalker enviroment
1. genetic.py
  * Inhouse written genetric Neural Network library code
1. genetic_gamer.py
  * interfaces with genetic.py for each of the enviroments. Can be called ''' python genetic_gamer.py ice|cart|walk ''' to run frozenLake, cartPole, or BipedalWalker enviroments respectively.
1. policy_gradient.py
  * policy gradient RL implementation used to work on the cartPole enviroment
1. rl_agent_nn.py
  * Q-Learning network based implementation to work on the FrozenLake Enviroment.
