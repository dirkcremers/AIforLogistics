import pandas as pd
import xlsxwriter
import numpy as np
from scipy.stats import gamma


def inventory_fill(mu: np.array, variance: np.array) -> np.array:
    """Calculates the inventory level to fill up to (rounded up)."""
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

    gamma_cdf = gamma.cdf(np.arange(0, 11), shape, scale=scale)
    gamma_cdf[-1] = 1.0
    descretized_demand = np.diff(gamma_cdf)
    descretized_demand = np.insert(descretized_demand, 0, gamma_cdf[0])
    return descretized_demand


def routing_cost(actions: np.array) -> float:
    """Calculates the routing cost for the given action.

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

    # initialize the value function
    V = np.zeros((11, 11, 11))

    # determine the inventory level to fill up to
    inventory_up_to = inventory_fill(mu, variance)

    # calculate the discrete demand distribution
    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    # Keep track of number of iterations of the policy evaluation algorithm
    iteration = 1

    # Run the policy evaluation algorithm until convergence
    while True:
        print(iteration)

        # Store the old value function to check for convergence
        V_old = V.copy()

        # Iterate over all possible states
        for store1 in range(11):
            for store2 in range(11):
                for store3 in range(11):
                    # Determine the action corresponding to the current state (knowing that we fill up to the inventory level)
                    actions = inventory_up_to - np.array([store1, store2, store3])
                    actions[actions < 0] = 0

                    # Calculate the value of the current state given the action for all the possible demand scenarios
                    value = 0
                    for demand1, prob1 in enumerate(store_1_demand):
                        for demand2, prob2 in enumerate(store_2_demand):
                            for demand3, prob3 in enumerate(store_3_demand):
                                prob = prob1 * prob2 * prob3

                                # Calculate the new state given the action and the demand
                                new_stores = inventory_up_to - np.array([demand1, demand2, demand3])

                                # Calculate the shortage and holding cost
                                shortage = -np.sum(new_stores[new_stores < 0])
                                holding = np.sum(new_stores[new_stores >= 0])

                                # set negative values to zero
                                new_stores[new_stores < 0] = 0
                                new_stores = new_stores.astype(int)

                                # Calculate the reward
                                reward = shortage * -19 + holding * -1 - routing_cost(actions)

                                # Calculate the value of the current state given the action and the probability of that given scenario
                                value += prob * (reward + GAMMA * V[new_stores[0], new_stores[1], new_stores[2]])

                    # Store the value of the current state in the value function
                    V[store1, store2, store3] = value

        # Check for convergence and increase iteration counter by 1
        iteration += 1
        if np.max(np.abs(V - V_old)) < 1:
            break

    # return the value function
    return V


def policy_improvement_single_step(GAMMA: float) -> np.array:

    # initialize the value function and the best action matrix
    best_actions = np.zeros((11, 11, 11), dtype=object)
    best_values = np.zeros((11, 11, 11))

    # obtain the Value function after policy evaluation
    V = policy_evaluation(GAMMA)

    # obtain the descretized demand distribution
    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    iteration = 1

    # Run the policy improvement algorithm only one iteration
    while iteration < 2:
        print(iteration)
        iteration += 1

        # Iterate over all possible states
        for store1 in range(11):
            for store2 in range(11):
                for store3 in range(11):

                    # set initial value of best value and best action
                    best_value = -np.inf
                    best_action = [-100, -100, -100]

                    # Iterate now over all possible actions to see if we can improve the policy with a certain action
                    for action1 in range(0, 11):
                        for action2 in range(0, 11):
                            for action3 in range(0, 11):
                                actions = np.array([action1, action2, action3])

                                # Calculate the value of the current state given the action for all the possible demand scenarios
                                value = 0
                                for demand1, prob1 in enumerate(store_1_demand):
                                    for demand2, prob2 in enumerate(store_2_demand):
                                        for demand3, prob3 in enumerate(store_3_demand):
                                            prob = prob1 * prob2 * prob3

                                            # Calculate the new state given the action and the demand
                                            # Also make sure that the new state is not larger than 10
                                            new_stores = np.array([min(store1 + action1, 10) - demand1,
                                                                   min(store2 + action2, 10) - demand2,
                                                                   min(store3 + action3, 10) - demand3])

                                            # Calculate the shortage and holding cost
                                            shortage = -np.sum(new_stores[new_stores < 0])
                                            holding = np.sum(new_stores[new_stores >= 0])

                                            # set negative values to zero
                                            new_stores[new_stores < 0] = 0
                                            new_stores = new_stores.astype(int)

                                            # Calculate the reward
                                            reward = shortage * -19 + holding * -1 - routing_cost(actions)

                                            # add up the value of the current state given the action and the probability of that given scenario
                                            value += prob * (
                                                    reward + GAMMA * V[new_stores[0], new_stores[1], new_stores[2]])

                                # save best action and best value until now
                                if value > best_value:
                                    best_value = value
                                    best_action = actions

                    # Store the best action and best value in the corresponding matrices
                    best_actions[store1, store2, store3] = best_action
                    best_values[store1, store2, store3] = best_value

    return best_actions, best_values


GAMMA = 0.8
mu = np.array([3, 5, 2])
variance = np.array([1, 2, 3])

# Question 5b
prob = descretized_demand_distribution(mu[0], variance[0])
print(prob)

V = policy_evaluation(GAMMA)
print(V)

# Question 5c
policy_new, V_new = policy_improvement_single_step(GAMMA)
pd.DataFrame(policy_new).to_csv('ValueIterationImprovement_policy.csv')
pd.DataFrame(V_new).to_csv('ValueIterationImprovement_value.csv')

# Question 5d

# parameters for Q-learning
n_episodes = 1000000
len_episode = 75
epsilon = 0.05
alpha = 0.01

def tabular_Q_learning(n_episodes: int, len_episode: int, epsilon: float, alpha: float) -> np.array:
    # initialize the Q function
    Q = np.zeros((11, 11, 11, 11, 11, 11))

    # obtain the descretized demand distribution
    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    # run the Q-learning algorithm for a certain amount of episodes
    for iteration in range(n_episodes):
        print(f'({iteration}/{n_episodes})')

        # start with random initial inventory for convergence purposes
        store1, store2, store3 = np.random.randint(0, 11, 3)

        # run the episode
        for _ in range(len_episode):

            # take action according to epsilon greedy
            if np.random.uniform(0, 1) < epsilon:
                action1, action2, action3 = np.random.randint(0, 11, 3)
            else:
                action1, action2, action3 = np.unravel_index(np.argmax(Q[store1, store2, store3, :, :, :]),
                                                             Q[store1, store2, store3, :, :, :].shape)

            # sample the demand
            demand1 = np.random.choice(np.arange(0, 11), p=store_1_demand)
            demand2 = np.random.choice(np.arange(0, 11), p=store_2_demand)
            demand3 = np.random.choice(np.arange(0, 11), p=store_3_demand)

            # calculate the new state (note again that the new state cannot be larger than 10)
            new_stores = np.array([min(store1 + action1, 10) - demand1, min(store2 + action2, 10) - demand2,
                                   min(store3 + action3, 10) - demand3])

            # calculate the shortage and holding cost
            shortage = -np.sum(new_stores[new_stores < 0])
            holding = np.sum(new_stores[new_stores >= 0])

            # set negative values to zero
            new_stores[new_stores < 0] = 0
            new_stores = new_stores.astype(int)

            # calculate the reward
            reward = shortage * -19 + holding * -1 - routing_cost(np.array([action1, action2, action3]))

            # update the Q function
            Q[store1, store2, store3, action1, action2, action3] += alpha * (
                        reward + GAMMA * np.max(Q[new_stores[0], new_stores[1], new_stores[2], :, :, :]) - Q[
                    store1, store2, store3, action1, action2, action3])

            # update the state
            store_1, store2, store3 = new_stores

    # return the Q function
    return Q

Q = tabular_Q_learning(n_episodes, len_episode, epsilon, alpha)

# obtain the optimal policy and the corresponding value function
# determined by the outcome of the Q function of the Q learning algorithm
policy = np.zeros((11, 11, 11), dtype=object)
V = np.zeros((11, 11, 11))
for store1 in range(11):
    for store2 in range(11):
        for store3 in range(11):
            policy[store1, store2, store3] = np.unravel_index(np.argmax(Q[store1, store2, store3, :, :, :]),
                                                              Q[store1, store2, store3, :, :, :].shape)
            V[store1, store2, store3] = np.max(Q[store1, store2, store3, :, :, :])


# Save results to excel
#
# # Policy
# writer = pd.ExcelWriter('Q_learning_policy.xlsx', engine='xlsxwriter')
# for i in range(0, 11):
#     df = pd.DataFrame(policy[:,:,i])
#     df.to_excel(writer, sheet_name='bin%d' % i)
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


# Question 6
n_episodes = 1000000
len_episode = 75
epsilon = 0.05
alpha = 0.01

def tabular_Q_learning_modified(n_episodes: int, len_episode: int, epsilon: float, alpha: float) -> np.ndarray:
    # initialize the Q function
    Q = np.zeros((21, 11, 21, 11))

    # obtain the descretized demand distribution
    store_1_demand = descretized_demand_distribution(mu[0], variance[0])
    store_2_demand = descretized_demand_distribution(mu[1], variance[1])
    store_3_demand = descretized_demand_distribution(mu[2], variance[2])

    # run the Q-learning algorithm for a certain amount of episodes
    for iteration in range(n_episodes):
        print(f'({iteration}/{n_episodes})')

        # start with random initial inventory to observe all states
        cluster1 = np.random.randint(0, 21)
        cluster2 = np.random.randint(0, 11)

        # run the episode
        for _ in range(len_episode):

            # take action according to epsilon greedy
            if np.random.uniform(0, 1) < epsilon:
                action1 = np.random.randint(0, 21)
                action2 = np.random.randint(0, 11)
            else:
                action1, action2 = np.unravel_index(np.argmax(Q[cluster1, cluster2, :, :]),
                                                             Q[cluster1, cluster2, :, :].shape)

            # sample the demand
            demand1 = np.random.choice(np.arange(0, 11), p=store_1_demand) + np.random.choice(np.arange(0, 11), p=store_2_demand)
            demand2 = np.random.choice(np.arange(0, 11), p=store_3_demand)

            # calculate the new state (note again that the new state cannot be larger than 10)
            new_stores = np.array([min(cluster1 + action1, 20) - demand1, min(cluster2 + action2, 10) - demand2])

            # calculate the shortage and holding cost
            shortage = -np.sum(new_stores[new_stores < 0])
            holding = np.sum(new_stores[new_stores >= 0])

            # set negative values to zero
            new_stores[new_stores < 0] = 0
            new_stores = new_stores.astype(int)

            # calculate the reward
            reward = shortage * -19 + holding * -1 - routing_cost(np.array([0, action1, action2]))

            # update the Q function
            Q[cluster1, cluster2, action1, action2] += alpha * (
                        reward + GAMMA * np.max(Q[new_stores[0], new_stores[1], :, :]) - Q[
                    cluster1, cluster2, action1, action2])

            # update the state
            cluster1, cluster2 = new_stores

    # return the Q function
    return Q

Q = tabular_Q_learning_modified(n_episodes, len_episode, epsilon, alpha)

# obtain the optimal policy and the corresponding value function
# determined by the outcome of the Q function of the Q learning algorithm modified
policy = np.zeros((21, 11), dtype=object)
V = np.zeros((21, 11))
for cluster1 in range(21):
    for cluster2 in range(11):
        policy[cluster1, cluster2] = np.unravel_index(np.argmax(Q[cluster1, cluster2, :, :]),
                                                          Q[cluster1, cluster2, :, :].shape)
        V[cluster1, cluster2] = np.max(Q[cluster1, cluster2, :, :])

# Save results to excel
#
# # Policy
# writer = pd.ExcelWriter('Q_learning_policy_6.xlsx', engine='xlsxwriter')
# df = pd.DataFrame(policy)
# df.to_excel(writer)
# writer.save()
#
# # Value function
# writer = pd.ExcelWriter('Q_learning_V_6.xlsx', engine='xlsxwriter')
# df = pd.DataFrame(V)
# df.to_excel(writer)
# writer.save()

