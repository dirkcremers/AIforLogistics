import numpy as np

from dirp.AI4LEnvironment import AI4LEnvironment

# sample action
action = np.random.randint(0, 8, size=20)
action[0] = 1
print(action)

# sample demand
inventory = np.random.randint(0, 100, size=20)
inventory[0] = 0

settings = {'transport_distance_factor': 1, 'transport_fixed_factor': 1, 'action_space': 8, 'routing_approx': False}
env = AI4LEnvironment(settings)

#
