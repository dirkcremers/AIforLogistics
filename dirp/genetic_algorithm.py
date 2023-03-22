import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import hygese as hgs


class GeneticAlgorithm:
    def __init__(self, env):

        self.env = env

        self.demand = self.env.generate_demand()
        self.inventory = self.env.inventories.copy()

    def run(self):

        var_bound = np.array([[0, 1]] * (self.env.nStores + 1))
        var_bound[0, 0] = 1

        model = ga(function=self.__cost_function,
                   dimension=self.env.nStores+1,
                   variable_type='int',
                   variable_boundaries=var_bound,
                   algorithm_parameters={'max_num_iteration': 100,
                                    'population_size': 100,
                                    'mutation_probability': 0.1,
                                    'elit_ratio': 0.01,
                                    'crossover_probability': 0.5,
                                    'parents_portion': 0.3,
                                    'crossover_type': 'uniform',
                                    'max_iteration_without_improv': None}
                   )

        model.run()
        return model.output_dict['variable'], model.output_dict['function']

    def __cost_function(self, action):

        # Create copies of the inventory and demand
        inventory = self.inventory.copy()
        demand = self.demand.copy()

        # TODO: do we need to choose action outside of the simulation part
        cost = 0

        # routing cost
        # TODO: check how we want to implement this
        # cost += self.__routing_cost(action)

        # add inventory which is deliverd based by the action
        inventory = inventory + (self.env.orderUpTo - inventory) * action

        # calculate holding cost and lost sales cost
        for i in range(0, self.env.nStores):
            inventory[i] -= demand[i]
            cost -= max(0, inventory[i]) * self.env.c_holding + -1 * min(0, inventory[i]) * self.env.c_lost

        # Algorithm tries to minimizes hence mulitply by -1
        return -cost


    def __routing_cost(self, action):
        self.data = self.env.data.copy()

        # Note demand for store
        self.data['demands'] = (self.env.orderUpTo - self.inventory) * action

        # Remove stores which are not visited
        no_visited_stores = np.where(np.array(action) == 0)[0]
        self.data['distance_matrix'] = np.delete(self.data['distance_matrix'], no_visited_stores, axis=0)
        self.data['distance_matrix'] = np.delete(self.data['distance_matrix'], no_visited_stores, axis=1)

        self.data['demands'] = np.delete(self.data['demands'], no_visited_stores)
        self.data['service_times'] = np.delete(self.data['service_times'], no_visited_stores)

        # Solve the routing problem
        ap = hgs.AlgorithmParameters(timeLimit=0.1)  # seconds
        hgs_solver = hgs.Solver(parameters=ap, verbose=False)

        result = hgs_solver.solve_cvrp(self.data)

        return -1 * result.cost