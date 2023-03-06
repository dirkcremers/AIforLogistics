import pandas as pd
import xlsxwriter
import numpy as np
from scipy.stats import gamma


def inventory_fill(mu: np.array, variance: np.array) -> np.array:
    inventory = mu + 1.97 * np.sqrt(variance)

    return np.ceil(inventory)


def descretized_demand_distribution(mu: float, variance: float):
    """Returns a discrete distribution of the demand.

    Args:
        mu (float): mean of the demand
        sigma (float): standard deviation of the demand

    Returns:
        np.ndarray: discrete distribution of the demand
    """

    scale = variance / mu
    shape = mu / scale

    a = gamma.cdf(np.arange(0, 11), shape, scale=scale)
    a[-1] = 1.0
    b = np.diff(a)
    b = np.insert(b, 0, a[0])
    return b


def routing_cost(actions: np.array) -> float:
    """Calculates the routing cost.

    Args:
        actions (np.array): actions to take

    Returns:
        float: routing cost
    """
    routes_to_drive = actions > 0
    if np.sum(routes_to_drive) == 3:
        return 500
    elif np.sum(abs(routes_to_drive - np.array([1, 1, 0]))) == 0:
        return 60
    elif np.sum(routes_to_drive) == 2:
        return 75
    elif np.sum(abs(routes_to_drive - np.array([0, 0, 1]))) == 0:
        return 55
    elif np.sum(routes_to_drive) == 1:
        return 40
    else:
        return 0


def policy_evaluation(GAMMA: float) -> np.array:
    V = np.zeros((11, 11, 11))

    inventory_up_to = inventory_fill(mu, variance)

    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    iteration = 1

    while True:
        print(iteration)

        V_old = V.copy()

        for store1 in range(11):
            for store2 in range(11):
                for store3 in range(11):
                    actions = inventory_up_to - np.array([store1, store2, store3])
                    actions[actions < 0] = 0

                    value = 0

                    for demand1, prob1 in enumerate(store_1_demand):
                        for demand2, prob2 in enumerate(store_2_demand):
                            for demand3, prob3 in enumerate(store_3_demand):
                                prob = prob1 * prob2 * prob3

                                new_stores = inventory_up_to - np.array([demand1, demand2, demand3])

                                shortage = -np.sum(new_stores[new_stores < 0])
                                holding = np.sum(new_stores[new_stores >= 0])

                                # set negative values to zero
                                new_stores[new_stores < 0] = 0
                                new_stores = new_stores.astype(int)

                                reward = shortage * -19 + holding * -1 - routing_cost(actions)

                                value += prob * (reward + GAMMA * V[new_stores[0], new_stores[1], new_stores[2]])

                    V[store1, store2, store3] = value

        iteration += 1
        if np.max(np.abs(V - V_old)) < 1:
            break

    return V


def policy_evaluation_single_step(GAMMA: float) -> np.array:

    best_actions = np.zeros((11, 11, 11))

    V = policy_evaluation(GAMMA)

    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    iteration = 1

    while iteration < 2:
        print(iteration)
        iteration += 1

        for store1 in range(11):
            for store2 in range(11):
                for store3 in range(11):

                    best_value = -np.inf
                    best_action = -10

                    for action1 in range(0, 11):
                        for action2 in range(0, 11):
                            for action3 in range(0, 11):
                                actions = np.array([action1, action2, action3])

                                value = 0

                                for demand1, prob1 in enumerate(store_1_demand):
                                    for demand2, prob2 in enumerate(store_2_demand):
                                        for demand3, prob3 in enumerate(store_3_demand):
                                            prob = prob1 * prob2 * prob3

                                            new_stores = np.array([min(store1 + action1, 10) - demand1,
                                                                   min(store2 + action2, 10) - demand2,
                                                                   min(store3 + action3, 10) - demand3])

                                            shortage = -np.sum(new_stores[new_stores < 0])
                                            holding = np.sum(new_stores[new_stores >= 0])

                                            # set negative values to zero
                                            new_stores[new_stores < 0] = 0
                                            new_stores = new_stores.astype(int)

                                            reward = shortage * -19 + holding * -1 - routing_cost(actions)

                                            value += prob * (
                                                    reward + GAMMA * V[new_stores[0], new_stores[1], new_stores[2]])

                                if value > best_value:
                                    best_value = value
                                    best_action = actions

                    best_actions[store1, store2, store3] = best_action

    return best_actions


GAMMA = 0.8
mu = np.array([3, 5, 2])
variance = np.array([1, 2, 3])

# prob = descretized_demand_distribution(mu[0], variance[0])
# print(prob)
#
# V = policy_evaluation(GAMMA)
# print(V)
#
# # Q = policy_evaluation_single_step(GAMMA)
# # print(Q)
# # pd.DataFrame(V).to_csv('ValueIteration.csv')
#
# # %%
#
# Q = np.zeros((11, 11, 11, 11, 11, 11))
# n_episodes = 1000000
# len_episode = 75
# epsilon = 0.05
# Q[10, 10, 10, 1, 2, 3] = 1
# alpha = 0.01
#
# store_1_demand = descretized_demand_distribution(mu[0], variance[0])
# store_2_demand = descretized_demand_distribution(mu[1], variance[1])
# store_3_demand = descretized_demand_distribution(mu[2], variance[2])
#
# for iteration in range(n_episodes):
#     print(f'({iteration}/{n_episodes})')
#     store1, store2, store3 = np.random.randint(0, 11, 3)  # start with random initial inventory for convergence purposes
#     for _ in range(len_episode):
#         # take action epsilon greedy
#         if np.random.uniform(0, 1) < epsilon:
#             action1, action2, action3 = np.random.randint(0, 11, 3)
#         else:
#             action1, action2, action3 = np.unravel_index(np.argmax(Q[store1, store2, store3, :, :, :]),
#                                                          Q[store1, store2, store3, :, :, :].shape)
#         demand1 = np.random.choice(np.arange(0, 11), p=store_1_demand)
#         demand2 = np.random.choice(np.arange(0, 11), p=store_2_demand)
#         demand3 = np.random.choice(np.arange(0, 11), p=store_3_demand)
#
#         new_stores = np.array([min(store1 + action1, 10) - demand1, min(store2 + action2, 10) - demand2,
#                                min(store3 + action3, 10) - demand3])
#
#         shortage = -np.sum(new_stores[new_stores < 0])
#         holding = np.sum(new_stores[new_stores >= 0])
#
#         # set negative values to zero
#         new_stores[new_stores < 0] = 0
#         new_stores = new_stores.astype(int)
#
#         reward = shortage * -19 + holding * -1 - routing_cost(np.array([action1, action2, action3]))
#
#         Q[store1, store2, store3, action1, action2, action3] += alpha * (
#                     reward + GAMMA * np.max(Q[new_stores[0], new_stores[1], new_stores[2], :, :, :]) - Q[
#                 store1, store2, store3, action1, action2, action3])
#
#         store_1, store2, store3 = new_stores
#
# policy = np.zeros((11, 11, 11), dtype=object)
# V = np.zeros((11, 11, 11))
# for store1 in range(11):
#     for store2 in range(11):
#         for store3 in range(11):
#             policy[store1, store2, store3] = np.unravel_index(np.argmax(Q[store1, store2, store3, :, :, :]),
#                                                               Q[store1, store2, store3, :, :, :].shape)
#             V[store1, store2, store3] = np.max(Q[store1, store2, store3, :, :, :])
#
#
# # Save results to excel
#
# # Policy
#
# writer = pd.ExcelWriter('Q_learning_policy.xlsx', engine='xlsxwriter')
#
# for i in range(0, 11):
#     df = pd.DataFrame(policy[:,:,i])
#     df.to_excel(writer, sheet_name='bin%d' % i)
#
# writer.save()
#
# # Value function
# writer = pd.ExcelWriter('Q_learning_V.xlsx', engine='xlsxwriter')
#
# for i in range(0, 11):
#     df = pd.DataFrame(V[:,:,i])
#     df.to_excel(writer, sheet_name='bin%d' % i)
#
# writer.save()


Q = np.zeros((21, 11, 21, 11))
n_episodes = 1000000
len_episode = 75
epsilon = 0.05
Q[20, 10, 1, 2] = 1
alpha = 0.01

store_1_demand = descretized_demand_distribution(mu[0], variance[0])
store_2_demand = descretized_demand_distribution(mu[1], variance[1])
store_3_demand = descretized_demand_distribution(mu[2], variance[2])

for iteration in range(n_episodes):
    print(f'({iteration}/{n_episodes})')
    cluster1 = np.random.randint(0, 21)
    cluster2 = np.random.randint(0, 11)  # start with random initial inventory for convergence purposes
    for _ in range(len_episode):
        # take action epsilon greedy
        if np.random.uniform(0, 1) < epsilon:
            action1 = np.random.randint(0, 21)
            action2 = np.random.randint(0, 11)
        else:
            action1, action2 = np.unravel_index(np.argmax(Q[cluster1, cluster2, :, :]),
                                                         Q[cluster1, cluster2, :, :].shape)
        demand1 = np.random.choice(np.arange(0, 11), p=store_1_demand) + np.random.choice(np.arange(0, 11), p=store_2_demand)
        demand2 = np.random.choice(np.arange(0, 11), p=store_3_demand)


        new_stores = np.array([min(cluster1 + action1, 20) - demand1, min(cluster2 + action2, 10) - demand2])

        shortage = -np.sum(new_stores[new_stores < 0])
        holding = np.sum(new_stores[new_stores >= 0])

        # set negative values to zero
        new_stores[new_stores < 0] = 0
        new_stores = new_stores.astype(int)

        reward = shortage * -19 + holding * -1 - routing_cost(np.array([0, action1, action2]))

        Q[cluster1, cluster2, action1, action2] += alpha * (
                    reward + GAMMA * np.max(Q[new_stores[0], new_stores[1], :, :]) - Q[
                cluster1, cluster2, action1, action2])

        cluster1, cluster2 = new_stores

policy = np.zeros((21, 11), dtype=object)
V = np.zeros((21, 11))
for cluster1 in range(21):
    for cluster2 in range(11):
        policy[cluster1, cluster2] = np.unravel_index(np.argmax(Q[cluster1, cluster2, :, :]),
                                                          Q[cluster1, cluster2, :, :].shape)
        V[cluster1, cluster2] = np.max(Q[cluster1, cluster2, :, :])

# Save results to excel

# Policy

writer = pd.ExcelWriter('Q_learning_policy_6.xlsx', engine='xlsxwriter')

df = pd.DataFrame(policy)
df.to_excel(writer)

writer.save()

# Value function
writer = pd.ExcelWriter('Q_learning_V_6.xlsx', engine='xlsxwriter')

df = pd.DataFrame(V)
df.to_excel(writer)

writer.save()

