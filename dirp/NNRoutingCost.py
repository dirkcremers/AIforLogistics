import keras.models
import pandas as pd
import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

from RouingCostEnv import RoutingCostEnv


settings = {'transport_distance_factor': 1, 'transport_fixed_factor': 1, 'action_space': 8, 'routing_approx': False}
env = RoutingCostEnv(settings)

settings2 = {'transport_distance_factor': 1, 'transport_fixed_factor': 1, 'action_space': 8, 'routing_approx': True}
env2 = RoutingCostEnv(settings2)

# create empty dataframe
# create empty array
# input = np.zeros(20)
# output = np.zeros(1)
#
#
# for _ in range(10000):
#     # sample action
#     action = np.random.randint(0, 8, size=20)
#     action[0] = 1
#
#     # sample demand
#     inventory = np.random.randint(0, 100, size=20)
#     inventory[0] = 0
#
#     routing_cost, orders = env.routing_cost(inventory, action)
#     model_transport = keras.model.load_model
#
#     # add orders to ordering in a new row
#     input = np.vstack((input, orders))
#     output = np.vstack((output, routing_cost))
#
# pd.DataFrame(input).to_csv('input_hygese.csv', index=False, header=False)
# pd.DataFrame(output).to_csv('output_hygese.csv', index=False, header=False)
#
# a = 1
# # create NN which with input the orders and as output the routing cost
# print(type(input))
# print(type(output))

input = pd.read_csv('input_hygese.csv', sep=',', header=None).values
output = pd.read_csv('output_hygese.csv', sep=',', header=None).values

# define the neural network model
# model = Sequential()
# model.add(Dense(40, input_shape=(20,), activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='linear'))
#
# # compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # train the model
# model.fit(input, output, epochs=800, verbose=1)
# model.save("NN_transportation_cost")

# load model
model = keras.models.load_model("NN_transportation_cost")


df = pd.DataFrame(
        columns=['Hygese', 'NN'])
# make a prediction
for i in range(1000):
    sample = np.random.randint(5, 34, size=20)
    num_to_zero = np.random.randint(0, 16)
    sample[np.random.choice(len(sample), size=num_to_zero, replace=False)] = 0
    print(sample)
    test = np.array([sample])
    test[0][0] = 0
    result = model.predict(test)
    true_value, _ = env.routing_cost(test[0])

    df = df.append(
        {'Hygese': true_value, 'NN': result},ignore_index=True)

df.to_csv(f'results_approx_nn01.csv', index=False)

# from keras.models import Sequential
# from keras.layers import Dense
#
# # define the neural network model
# model = Sequential()
# model.add(Dense(10, input_dim=20, activation='relu'))
# model.add(Dense(1, activation='linear'))
#
# # compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # define the input and output data
# X = [[0, 12, 11]]
# Y = [-100]
#
# # train the model
# model.fit(X, Y, epochs=1000, verbose=0)
#
# # make a prediction
# result = model.predict([[0, 12, 11]])
# print(result)
