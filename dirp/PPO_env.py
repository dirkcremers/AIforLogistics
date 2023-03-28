import gym
import pandas as pd
from gym import spaces
import numpy as np
import geopy.distance
from stable_baselines3 import PPO

from dirp.AI4LEnvironment import AI4LEnvironment


class PPO_env(gym.Env):
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

        # print("Inventory size upon constr: ", len(self.inventories))

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
        self.orderUpTo = 2 * np.ceil(self.demandMean + 1.96 * np.sqrt(self.demandStdev))

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

        # StableBaseLines requires these else gives error if they are not defined
        self.spec = None
        self.metadata = None

    def __routing_cost(self, action):

        # fix depot to be always visited
        routing_action = action.copy()
        routing_action[0] = 1

        # Heuristic approach to determine cost of transportation

        # Optimal routing for the given problem (visiting every store | one truck)
        full_route = [0, 11, 12, 13, 14, 10, 9, 18, 19, 16, 17, 15, 7, 8, 6, 4, 5, 3, 2, 1, 0]

        # Order amounts which are delivered by the trucks
        orders = (self.orderUpTo - self.inventories) * routing_action

        # Indexes of alle stores which are not visited by our action
        stores_not_visited = [i for i, x in enumerate(routing_action) if x == 0]

        # Obtain approximation route by removing all stores which are not visited
        current_route = [x for x in full_route if x not in stores_not_visited]
        cost = 0
        for i in range(0, len(current_route)-1):
            cost += self.data['distance_matrix'][current_route[i], current_route[i+1]]

        # cost of the amount of trucks necessary
        cost += self.fixedTransportCost * np.ceil((np.sum(orders) / self.capacity))

        # print('Heuristic approach: ', cost)

        return -1 * cost
    def generate_demand(self):
        # generate random demand
        demands = np.zeros(self.nStores+1)

        for i in range(1, self.nStores+1):
            demands[i] = int(max(0, np.random.normal(self.demandMean[i], self.demandStdev[i])))

        return demands

    def step(self, action):

        # Create demand and initial cost
        demand = self.generate_demand()
        self.current_step += 1

        # routing cost
        reward = self.__routing_cost(action)

        # add inventory which is deliverd based by the action
        self.inventories = self.inventories + (self.orderUpTo - self.inventories) * action - demand

        # calculate holding cost and lost sales cost
        for i in range(0, self.nStores + 1):
            reward -= max(0, self.inventories[i]) * self.c_holding + -1 * min(0, self.inventories[i]) * self.c_lost

        # update inventory such that there is no negative inventory
        self.inventories = np.maximum(self.inventories, 0)

        self.cost += reward
        self.avgCost = self.cost / self.current_step
        done = self.current_step >= 2000
        obs = self.inventories

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.inventories = np.zeros(self.nStores + 1)

        self.current_step = 0

        self.cost = 0
        self.avgCost = 0

        return self.inventories

    def render(self, mode='human', close=False):
        print("No rendering implemented")

env = PPO_env()
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=2000000)
model.save("ppo_truck")

del model
env.close()
model = PPO.load("ppo_truck")


# create the environment
env_test = AI4LEnvironment()

df = pd.DataFrame(columns=['iteration', 'inventory', 'action', 'reward', 'avgCost', 'HoldCost', 'LostCost', 'RoutingCost'])

# reset the environment
obs = env_test.reset()
iteration = 0
done = False

while done == False:

    # copy inventory before demand and action are taken
    obs_old = obs.copy()

    print('\n', iteration, '------------------------------------')
    action, _states = model.predict(obs, deterministic=True)
    action[0] = 1

    # take the action
    obs, reward, done, info = env_test.step(action)
    print('inventory: ', obs)
    print('reward: ', reward)
    print('average cost: ', env_test.avgCost)
    iteration += 1

    # save results
    df = df.append(
        {'iteration': iteration, 'inventory': obs_old, 'action': action, 'reward': reward, 'avgCost': env_test.avgCost,
         'HoldCost': info['HoldingCost'], 'LostCost': info['LostCost'], 'RoutingCost': info['TransportationCost']},
        ignore_index=True)

df.to_csv('results_ppo.csv', index=False)

