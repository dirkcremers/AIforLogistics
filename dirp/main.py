import numpy as np

from dirp.AI4LEnvironment import AI4LEnvironment
from dirp.genetic_algorithm import GeneticAlgorithm

# create the environment
env = AI4LEnvironment()

# reset the environment
obs = env.reset()
iteration = 0
done = False

while done == False:
    print('\n', iteration, '------------------------------------')

    # create the genetic algorithm
    gen = GeneticAlgorithm(env)
    action = gen.run()

    # SS policy
    # action = np.less_equal(obs, env.s).astype(int)
    # action[0] = 1
    # print(action)

    # take the action
    obs, reward, done, info = env.step(action)
    print('inventory: ', obs)
    print('reward: ', reward)
    print('average cost: ', env.avgCost)



    iteration += 1
