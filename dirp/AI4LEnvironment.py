import gym
from gym import spaces
import numpy as np
import hygese as hgs
import geopy.distance

class AI4LEnvironment(gym.Env):
    """Joint Replenishment Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.nStores = 19

        self.data = dict()
        self.lat = np.array([51.46455753,  # arbitrarily selected as depot
                             51.48315348,
                             51.47946524,
                             51.47989288,
                             51.47091165,
                             51.46823832,
                             51.45097352,
                             51.44236109,
                             51.44075611,
                             51.43310493,
                             51.42748611,
                             51.44209359,
                             51.41860162,
                             51.41196388,
                             51.41919040,
                             51.45204327,
                             51.47407462,
                             51.46455753,
                             51.41655806,
                             51.46071441
                             ])

        self.lon = np.array([5.441001338,  # arbitrarily selected as depot
                             5.449615716,
                             5.463520286,
                             5.478883978,
                             5.463348625,
                             5.47776818,
                             5.465205594,
                             5.47979681,
                             5.471642895,
                             5.489753169,
                             5.465033933,
                             5.408385679,
                             5.407870695,
                             5.409587309,
                             5.441001338,
                             5.517476483,
                             5.546744746,
                             5.550177973,
                             5.572865385,
                             5.609406608
                             ])

        self.data['distance_matrix'] = np.zeros(shape=[self.nStores + 1, self.nStores + 1])
        self.transportCost = 2.5 * self.settings['transport_distance_factor']
        self.fixedTransportCost = 20 * self.settings['transport_fixed_factor']

        for i in range(0, self.nStores + 1):
            for j in range(0, self.nStores + 1):
                coords_1 = (self.lat[i], self.lon[i])
                coords_2 = (self.lat[j], self.lon[j])

                self.data['distance_matrix'][i][j] = geopy.distance.geodesic(coords_1, coords_2).km * self.transportCost

                if i == 0:
                    self.data['distance_matrix'][i][j] += self.fixedTransportCost
        self.data['service_times'] = np.zeros(self.nStores + 1)

        # the vehicle capacity, the number of vehicles, and depot information
        # this is stored in the 'data' dict to be compatible with hygese solver
        self.data['vehicle_capacity'] = 100
        self.data['num_vehicles'] = 19
        self.data['depot'] = 0

        # Information of the stores
        # TODO: make holding cost compatible with settings dictionary
        self.c_holding = 1
        self.c_lost = 19
        self.capacity = 1000

        # The maximum to be shipped to a store
        self.maxOrderQuantity = 100

        # the current amount of inventory in each store
        self.inventories = np.zeros(self.nStores + 1)

        # information on the demand distribution
        # small, medium or large stores: 4, 10, 25 shape par.

        self.demandMean = np.array([0,  # fake depot entry to let indices match
                                    10,
                                    4,
                                    10,
                                    10,
                                    25,
                                    25,
                                    4,
                                    25,
                                    25,
                                    25,
                                    4,
                                    10,
                                    4,
                                    10,
                                    10,
                                    4,
                                    4,
                                    10,
                                    4])

        np.random.seed(1331)

        # Standard deviation
        self.demandStdev = np.ceil(np.random.rand(self.nStores + 1) * 0.5 * self.demandMean)

        # For bookkeeping purposes
        self.demands = np.zeros(self.nStores + 1)
        self.action = np.zeros(self.nStores + 1)

        self.cost = 0
        self.avgCost = 0

        self.holdingCost = 0
        self.avgHoldingCost = 0

        self.lostCost = 0
        self.avgLostCost = 0

        self.transportCost = 0
        self.avgTransportCost = 0

        self.reward_range = (self.nStores * -1 * self.capacity * self.c_lost, 3 * self.capacity * self.c_holding)

        # action space
        self.action_space = spaces.MultiDiscrete([self.settings['action_space']] * (self.nStores + 1))

        # observation space is simply the inventory levels at each store at the
        # start of the day
        self.observation_space = spaces.Box(low=0,
                                            high=self.capacity,
                                            shape=(self.nStores + 1,),
                                            dtype=np.int32)

        # StableBaseLines requires these else gives error if they are not defined
        self.spec = None
        self.metadata = None

    def __routing_cost(self, action):

        # Fix depot to be always visited
        orderUpTo = np.ceil(self.demandMean * action + 1.96 * np.sqrt(action) * self.demandStdev)
        self.data['demands'] = np.where(orderUpTo - self.inventories > self.maxOrderQuantity, self.maxOrderQuantity, orderUpTo - self.inventories)
        self.data['demands'] = np.where(orderUpTo >= self.capacity,
                                        self.capacity - self.inventories, self.data['demands'])
        self.data['demands'] = np.where(orderUpTo - self.inventories < 0, 0, self.data['demands'])

        action = np.where(self.data['demands'] > 0, action, 0)
        action[0] = 1

        # for bookkeeping purposes
        self.action_taken = action
        self.order_shipped = self.data['demands']

        # if not trucks need to drive, then cost = 0
        if np.sum(action[1:]) == 0:
            return 0

        # Heuristic approach to determine cost of transportation
        if self.settings['routing_approx']:

            # Optimal routing for the given problem (visiting every store | one truck)
            full_route = [0, 11, 12, 13, 14, 10, 9, 18, 19, 16, 17, 15, 7, 8, 6, 4, 5, 3, 2, 1, 0]

            # Indexes of alle stores which are not visited by our action
            stores_not_visited = [i for i, x in enumerate(action) if x == 0]

            # Obtain approximation route by removing all stores which are not visited
            current_route = [x for x in full_route if x not in stores_not_visited]
            cost = 0
            for i in range(0, len(current_route) - 1):
                cost += self.data['distance_matrix'][current_route[i], current_route[i + 1]]

            # cost of the amount of trucks necessary
            cost += self.fixedTransportCost * np.ceil((np.sum(self.data['demands']) / self.data['vehicle_capacity']))

            # print('Heuristic approach: ', cost)

            return -1 * cost

        # Hygese best solution
        else:
            if np.sum(np.where(action[1:] > 0, 1, 0)) == 1:
                store_index = np.where(np.array(action) > 0)[0][1]

                no_trucks = np.ceil(self.data['demands'][store_index] / self.data['vehicle_capacity']).astype(int)
                routing_cost = self.data['distance_matrix'][0][store_index] + self.data['distance_matrix'][store_index][0]

                return -1 * (routing_cost * no_trucks)

            # modify the problem such that the routing cost for all the
            # stores are only used for the stores which are getting replenished
            # therefore create copy of the data
            data_vrp = self.data.copy()

            no_visited_stores = np.where(np.array(action) == 0)[0]

            data_vrp['distance_matrix'] = np.delete(data_vrp['distance_matrix'], no_visited_stores, axis=0)
            data_vrp['distance_matrix'] = np.delete(data_vrp['distance_matrix'], no_visited_stores, axis=1)

            data_vrp['demands'] = np.delete(data_vrp['demands'], no_visited_stores)
            data_vrp['service_times'] = np.delete(data_vrp['service_times'], no_visited_stores)

            ap = hgs.AlgorithmParameters(timeLimit=0.1)  # seconds
            hgs_solver = hgs.Solver(parameters=ap, verbose=False)

            result = hgs_solver.solve_cvrp(data_vrp)

            return -1 * result.cost
    def generate_demand(self):
        # generate random demand
        demands = np.zeros(self.nStores+1)

        for i in range(1, self.nStores+1):
            demands[i] = int(max(0, np.random.normal(self.demandMean[i], self.demandStdev[i])))

        return demands

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1

        # transportation cost
        transportation_cost = self.__routing_cost(action)

        # update the inventory based on the arriving orders
        self._take_action()

        # generate demand
        demands = self.generate_demand()

        # keep track of cost
        holding_cost = 0
        lost_cost = 0

        for i in range(0, self.nStores+1):
            self.inventories[i] -= demands[i]

            # holding cost
            holding_cost -= max(0, self.inventories[i]) * self.c_holding

            # lost sales cost
            lost_cost -= -1 * min(0, self.inventories[i]) * self.c_lost

            self.inventories[i] = max(0, self.inventories[i])

        total_cost = transportation_cost + holding_cost + lost_cost
        self.cost += total_cost
        self.avgCost = self.cost / self.current_step

        done = self.current_step >= 372

        obs = self._next_observation()

        return obs, total_cost, done, {'TransportationCost': transportation_cost,
                                       'HoldingCost': holding_cost,
                                       'LostCost': lost_cost,
                                       'action': self.action_taken,
                                       'shipped': self.order_shipped}

    def _take_action(self):

        # in this example it is rather simple; the inventory is shipped
        self.inventories = self.inventories + self.data['demands']

    def reset(self):
        # Reset the state of the environment to an initial state
        self.inventories = np.zeros(self.nStores + 1)

        self.current_step = 0

        self.cost = 0
        self.avgCost = 0

        return self._next_observation()

    def _next_observation(self):
        return self.inventories

    def render(self, mode='human', close=False):
        print("No rendering implemented")

