import numpy as np
import pandas as pd

from dirp.AI4LEnvironment import AI4LEnvironment
from dirp.genetic_algorithm import GeneticAlgorithm

# create dataframe for saving results
df = pd.DataFrame(columns=['iteration', 'inventory', 'action', 'reward', 'avgCost', 'HoldCost', 'LostCost', 'RoutingCost'])

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

    # copy inventory before demand and action are taken
    obs_old = obs.copy()

    # take the action
    obs, reward, done, info = env.step(action)
    print('inventory: ', obs)
    print('reward: ', reward)
    print('average cost: ', env.avgCost)

    # save results
    df = df.append({'iteration': iteration, 'inventory': obs_old, 'action': action, 'reward': reward, 'avgCost': env.avgCost, 'HoldCost': info['HoldingCost'], 'LostCost': info['LostCost'], 'RoutingCost': info['TransportationCost']}, ignore_index=True)

    iteration += 1


df.to_csv('results_genetic.csv', index=False)
