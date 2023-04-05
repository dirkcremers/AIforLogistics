import numpy as np
import pandas as pd

from dirp.RouingCostEnv import RoutingCostEnv


settings = {'transport_distance_factor': 1, 'transport_fixed_factor': 1, 'action_space': 8, 'routing_approx': False}
env = RoutingCostEnv(settings)

# create empty dataframe
df = pd.DataFrame(columns=['action', 'inventory', 'orders', 'routing_cost'])

# create empty array
input = np.zeros(20)
output = np.zeros(1)


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

    # add orders to ordering in a new row
    input = np.vstack((input, orders))
    output = np.vstakc((output, routing_cost))


a = 1
# create NN which with input the orders and as output the routing cost

from keras.models import Sequential
from keras.layers import Dense

# define the neural network model
model = Sequential()
model.add(Dense(10, input_dim=20, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# define the input and output data
X = [[0, 12, 11]]
Y = [-100]

# train the model
model.fit(X, Y, epochs=1000, verbose=0)

# make a prediction
result = model.predict([[0, 12, 11]])
print(result)
