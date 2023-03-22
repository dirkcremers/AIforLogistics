import gym
from gym import spaces
import numpy as np
import hygese as hgs
import geopy.distance

from dirp.genetic_algorithm import GeneticAlgorithm


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
        self.transportCost = 2.5
        self.fixedTransportCost = 20

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
        self.orderUpTo = np.ceil(self.demandMean + 1.96 * np.sqrt(self.demandStdev))

        # For bookkeeping purposes
        self.demands = np.zeros(self.nStores + 1)
        self.action = np.zeros(self.nStores + 1)
        self.cost = 0
        self.avgCost = 0

        # OPEN AI GYM elements that need to be set
        # this should indicate between which values the rewards could fluctuate
        # (Your teacher has no real clue what happens with it)
        self.reward_range = (self.nStores * -1 * self.capacity * self.c_lost, 3 * self.capacity * self.c_holding)

        # we need to define the shape of an action
        # for this example, we set it equal to a simple multibinairy action
        # space. (series of zeros and ones for ordering or not)
        # It is quite crucial to understand the spaces objects. Please google!

        # Also note that this action is ignored as we use a base stock
        # a first step towards implementation could be to ignore visiting 
        # a store.

        # how many stores we will replenish to base stock?
        self.action_space = spaces.MultiBinary(self.nStores + 1)

        # observation space is simply the inventory levels at each store at the
        # start of the day
        self.observation_space = spaces.Box(low=0,
                                            high=self.capacity,
                                            shape=(self.nStores + 1,),
                                            dtype=np.int32)

    def calcDirectReward(self, action):

        self.data['demands'] = (self.orderUpTo - self.inventories) * action

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
        demands = np.zeros(self.nStores)

        for i in range(0, self.nStores):
            demands[i] = int(max(0, np.random.normal(self.demandMean[i], self.demandStdev[i])))

        return demands

    def step(self, action):
        # Execute one time step within the environment

        reward = self.calcDirectReward(action)

        self._take_action()
        self.current_step += 1

        demands = self.generate_demand()

        for i in range(0, self.nStores):
            self.inventories[i] -= demands[i]
            reward -= max(0, self.inventories[i]) * self.c_holding + -1 * min(0, self.inventories[i]) * self.c_lost

            self.inventories[i] = max(0, self.inventories[i])

        self.cost += reward
        self.avgCost = self.cost / self.current_step

        done = self.current_step >= 2000

        obs = self._next_observation()

        return obs, reward, done, {}

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

