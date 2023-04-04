import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import hygese as hgs
from sklearn.cluster import KMeans


class GeneticAlgorithm:
    def __init__(self, env, settings):

        self.env = env
        self.settings = settings

        self.demand = None
        self.inventory = self.env.inventories.copy()

    def run(self):

        n_samples = 50

        var_bound = np.array([[0, self.settings['action_space'] - 1]] * (self.env.nStores + 1))
        var_bound[0, 0] = 1

        # create matrix for each sample of demand and given action
        action_sample_matrix = np.zeros((n_samples, self.env.nStores + 1))

        for demand_sample in range(n_samples):
            self.demand = self.env.generate_demand()

            model = ga(function=self.__cost_function,
                       dimension=self.env.nStores + 1,
                       variable_type='int',
                       variable_boundaries=var_bound,
                       convergence_curve=False,
                       algorithm_parameters={'max_num_iteration': 25,
                                             'population_size': 50,
                                             'mutation_probability': 0.1,
                                             'elit_ratio': 0.01,
                                             'crossover_probability': 0.5,
                                             'parents_portion': 0.3,
                                             'crossover_type': 'uniform',
                                             'max_iteration_without_improv': None}
                       )

            model.run()
            action, cost = model.output_dict['variable'], model.output_dict['function']
            action_sample_matrix[demand_sample, :] = action

        action = self.__consensus(action_sample_matrix)
        print('optimal action: ', action)

        return action

    def __cost_function(self, action):

        # print('Action: ', action)

        # Create copies of the inventory and demand
        inventory = self.inventory.copy()
        demand = self.demand.copy()

        # Order amounts which are delivered by the trucks
        self.orderUpTo = np.ceil(self.env.demandMean * action + 1.96 * np.sqrt(action) * self.env.demandStdev)
        self.orders = np.where(self.orderUpTo - self.inventory > self.env.maxOrderQuantity, self.env.maxOrderQuantity,
                               self.orderUpTo - self.inventory)
        self.orders = np.where(self.orderUpTo >= self.env.capacity,
                               self.env.capacity - self.inventory, self.orders)
        self.orders = np.where(self.orderUpTo - self.inventory < 0, 0, self.orders)

        cost = 0

        # routing cost
        cost += self.__routing_cost(action)
        # print('Routing cost: ', cost)

        # add inventory which is deliverd based by the action
        inventory = inventory + self.orders

        # calculate holding cost and lost sales cost
        for i in range(0, self.env.nStores + 1):
            inventory[i] -= demand[i]
            cost -= max(0, inventory[i]) * self.env.c_holding + -1 * min(0, inventory[i]) * self.env.c_lost

        # Algorithm tries to minimizes hence mulitply by -1
        return -cost

    def __routing_cost(self, action):

        # Heuristic approach to determine cost of transportation

        # Optimal routing for the given problem (visiting every store | one truck)
        full_route = [0, 11, 12, 13, 14, 10, 9, 18, 19, 16, 17, 15, 7, 8, 6, 4, 5, 3, 2, 1, 0]

        # Indexes of alle stores which are not visited by our action
        stores_not_visited = [i for i, x in enumerate(action) if x == 0]

        # Obtain approximation route by removing all stores which are not visited
        current_route = [x for x in full_route if x not in stores_not_visited]
        cost = 0
        for i in range(0, len(current_route) - 1):
            cost += self.env.data['distance_matrix'][current_route[i], current_route[i + 1]]

        # cost of the amount of trucks necessary
        cost += self.env.fixedTransportCost * np.ceil((np.sum(self.orders) / self.env.data['vehicle_capacity']))

        # print('Heuristic approach: ', cost)

        return -1 * cost

    def __consensus(self, action_sample_matrix):

        # Option 1
        action = np.mean(action_sample_matrix, axis=0) + 0.2
        # set all values to 0 or 1 based on value being greater or smaller than 0.3
        action = np.round(action)

        # Option 2
        # action = None
        #
        # kmeans = KMeans(n_init=10, n_clusters=1, random_state=0).fit(action_sample_matrix)
        # centriod = kmeans.cluster_centers_[0]
        # # action = np.round(centriod)
        #
        # min_distance = np.inf
        # for i in range(0, action_sample_matrix.shape[0]):
        #     distance = np.linalg.norm(action_sample_matrix[i, :] - centriod)
        #     if distance < min_distance:
        #         min_distance = distance
        #         action = action_sample_matrix[i, :]

        return action
