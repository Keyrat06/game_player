import gym
import sys
import genetic


if __name__ == "__main__":
    game = "walk"
    if len(sys.argv) < 2:
        pass
    elif sys.argv[1] == "ice":
        game = "ice"
    elif sys.argv[1] == "cart":
        game = "cart"
    else:
        game = "walk"

    if game == "walk":
        POPULATION_SIZE = 100
        KEEP_PORPORTION = 0.25
        INPUT_SIZE = 24
        OUTPUT_SIZE = 4
        HIDDENS = [10, 10]
        GOAL_STEPS = 500
        GENERATIONS = 100
        env = gym.make('BipedalWalker-v2')
        env.reset()
        population = genetic.get_initial_population(POPULATION_SIZE, INPUT_SIZE, HIDDENS, OUTPUT_SIZE, net_type=genetic.Genetic_Net_continous)
        population = genetic.train(population, GENERATIONS, KEEP_PORPORTION, POPULATION_SIZE, env, net_type=genetic.Genetic_Net_continous, game_steps=GOAL_STEPS)

    if game == "cart":
        POPULATION_SIZE = 100
        KEEP_PORPORTION = 0.25
        INPUT_SIZE = 4
        OUTPUT_SIZE = 2
        HIDDENS = [10, 10]
        GOAL_STEPS = 500
        GENERATIONS = 100
        env = gym.make('CartPole-v1')
        env.reset()
        population = genetic.get_initial_population(POPULATION_SIZE, INPUT_SIZE, HIDDENS, OUTPUT_SIZE, net_type=genetic.Genetic_Net_discrete)
        population = genetic.train(population, GENERATIONS, KEEP_PORPORTION, POPULATION_SIZE, env, net_type=genetic.Genetic_Net_discrete, game_steps=GOAL_STEPS)

    if game == "ice":
        POPULATION_SIZE = 100
        KEEP_PORPORTION = 0.25
        INPUT_SIZE = 16
        OUTPUT_SIZE = 4
        HIDDENS = [50]
        GOAL_STEPS = 500
        GENERATIONS = 1000
        env = gym.make('FrozenLake-v0')
        env.reset()
        population = genetic.get_initial_population(POPULATION_SIZE, INPUT_SIZE, HIDDENS, OUTPUT_SIZE, net_type=genetic.Genetic_Net_discrete)
        population = genetic.train(population, GENERATIONS, KEEP_PORPORTION, POPULATION_SIZE, env, net_type=genetic.Genetic_Net_discrete, game_steps=GOAL_STEPS)

