from dirp.AI4LEnvironment import AI4LEnvironment
from dirp.genetic_algorithm import GeneticAlgorithm

# create the environment
env = AI4LEnvironment()

# reset the environment
env.reset()
done = False

while done == False:
    gen = GeneticAlgorithm(env)
    solution, cost = gen.run()

    # take the action
    obs, reward, done, info = env.step(solution)
    print(obs, reward, done, info)
