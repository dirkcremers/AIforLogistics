import gym
from gym import spaces
import numpy as np
import hygese as hgs
import geopy.distance

class AI4LEnvironment(gym.Env):
    """Joint Replenishment Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
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
        self.transportCost = 2.5 * 2
        self.fixedTransportCost = 20 * 2

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
        self.c_holding = 1
        self.c_lost = 19
        self.capacity = 1000

        # The maximum to be shipped to a store
        self.maxOrderQuantity = 1000

        # the current amount of inventory in each store
        self.inventories = np.zeros(self.nStores + 1)

        print("Inventory size upon constr: ", len(self.inventories))

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
        self.demandStdev = np.ceil(np.random.rand(self.nStores + 1) * 0.5 * self.demandMean)

        # create some fixed order up to levels
        self.s = np.ceil(self.demandMean + 1.96 * np.sqrt(self.demandStdev))

        self.orderUpTo = self.s

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

        # how many stores we will replenish to base stock?
        self.action_space = spaces.MultiDiscrete([4] * (self.nStores + 1))

        # observation space is simply the inventory levels at each store at the
        # start of the day
        self.observation_space = spaces.Box(low=0,
                                            high=self.capacity,
                                            shape=(self.nStores + 1,),
                                            dtype=np.int32)

    def calcDirectReward(self, action):

        self.data['demands'] = np.where(self.orderUpTo * action - self.inventories < 0, 0, self.orderUpTo * action - self.inventories)

        if np.sum(action[1:]) == 0:
            return 0

        if np.sum(np.where(action[1:] > 0, 1, 0)) == 1:
            store_index = np.where(np.array(action) == 1)[0][1]

            no_trucks = np.ceil(self.data['demands'][store_index] / self.data['vehicle_capacity']).astype(int)
            routing_cost = self.data['distance_matrix'][0][store_index] + self.data['distance_matrix'][store_index][0]

            return -1 * (routing_cost * no_trucks)

        # modify the problem such that the routing cost for all the
        # stores are only used for the stores which are getting replenished

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

        print("Action: ", action)
        print("Start routing cost")

        transportation_cost = self.calcDirectReward(action)

        self._take_action()

        print("Start holding cost / lost sales cost")
        demands = self.generate_demand()

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
                                       'LostCost': lost_cost}

    def _take_action(self):

        # in this example it is rather simple; the inventory is shipped
        self.inventories = self.inventories + self.data['demands']

    def reset(self):
        # Reset the state of the environment to an initial state
        self.inventories = np.zeros(self.nStores + 1)

        self.current_step = 0

        self.cost = 0
        self.avgCost = 0;

        return self._next_observation()

    def _next_observation(self):
        return self.inventories

    def render(self, mode='human', close=False):
        print("No rendering implemented")

