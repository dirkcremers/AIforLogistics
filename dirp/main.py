from dirp.AI4LEnvironment import AI4LEnvironment
from dirp.genetic_algorithm import GeneticAlgorithm

# create the environment
env = AI4LEnvironment()

# reset the environment
env.reset()
iteration = 0
done = False

while done == False:
    print('\n', iteration, '------------------------------------')
    gen = GeneticAlgorithm(env)
    solution, cost = gen.run()

    # take the action
    obs, reward, done, info = env.step(solution)
    print('inventory: ', obs)
    print('reward: ', reward)
    print('average cost: ', env.avgCost)
    iteration += 1
