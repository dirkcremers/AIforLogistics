import numpy as np
import pandas as pd

from dirp.RouingCostEnv import RoutingCostEnv


settings = {'transport_distance_factor': 1, 'transport_fixed_factor': 1, 'action_space': 8, 'routing_approx': False}
env = RoutingCostEnv(settings)

# create empty dataframe
df = pd.DataFrame(columns=['action', 'inventory', 'orders', 'routing_cost'])


for _ in range(10):
    # sample action
    action = np.random.randint(0, 8, size=20)
    action[0] = 1
    print(action)

    # sample demand
    inventory = np.random.randint(0, 100, size=20)
    inventory[0] = 0

    routing_cost, orders = env.routing_cost(action, inventory)

    df = df.append({'action': action, 'inventory': inventory, 'orders': orders, 'routing_cost': routing_cost}, ignore_index=True)

a = 1


# create NN which has input
