import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from AI4LEnvironment import AI4LEnvironment
from PPO_util import linear_schedule
from genetic_algorithm import GeneticAlgorithm


def run_simulation(policy: str, settings: dict) -> None:
    # create dataframe for saving results
    df = pd.DataFrame(
        columns=['iteration', 'inventory', 'action', 'shipped', 'reward', 'avgCost', 'HoldCost', 'LostCost', 'RoutingCost'])

    # create the environment
    settings['routing_approx'] = False
    env = AI4LEnvironment(settings)

    # reset the environment
    obs = env.reset()
    iteration = 0
    done = False

    while done == False:
        print('\n', iteration, '------------------------------------')

        # SS policy
        if policy == 'SS':
            S = env.demandMean + 1.96 * env.demandStdev
            action = np.less_equal(obs, S).astype(int)
            action[0] = 1

        # Genetic Algorithm policy
        if policy == 'GA':
            gen = GeneticAlgorithm(env, settings)
            action = gen.run()
            action[0] = 1

        # PPO policy
        if policy == 'PPO':
            if iteration == 0:
                settings['routing_approx'] = True
                env_PPO = AI4LEnvironment(settings)

                model = PPO('MlpPolicy', env_PPO, gamma=0.95, learning_rate=linear_schedule(0.001), verbose=0)
                model.learn(total_timesteps=2000000)
                model.save("ppo_dirp")
                settings['routing_approx'] = False

            action, _states = model.predict(obs, deterministic=False)
            # ensure depot is always open
            action[0] = 1

        # copy inventory before demand and action are taken
        obs_old = obs.copy()

        # take the action
        obs, reward, done, info = env.step(action)
        print('inventory: ', obs)
        print('reward: ', reward)
        print('average cost: ', env.avgCost)

        # save results
        df = df.append(
            {'iteration': iteration, 'inventory': obs_old, 'action': action, 'reward': reward, 'avgCost': env.avgCost,
             'HoldCost': info['HoldingCost'], 'LostCost': info['LostCost'], 'RoutingCost': info['TransportationCost']},
            ignore_index=True)

        iteration += 1

    no_actions = settings['action_space']
    transportation_factor = settings['transport_distance_factor']

    df.to_csv(f'results_{policy}_{no_actions}_{transportation_factor}.csv', index=False)


if __name__ == '__main__':

    # Marnik
    # note that actual actions are: action_space - 1
    # settings = {'transport_distance_factor': 1,
    #             'transport_fixed_factor': 1,
    #             'action_space': 2}
    #
    # run_simulation('SS', settings)
    # run_simulation('GA', settings)
    # run_simulation('PPO', settings)

    # settings = {'transport_distance_factor': 1,
    #             'transport_fixed_factor': 1,
    #             'action_space': 4}
    # run_simulation('GA', settings)

    # settings = {'transport_distance_factor': 1,
    #             'transport_fixed_factor': 1,
    #             'action_space': 8}
    # run_simulation('PPO', settings)

    # settings = {'transport_distance_factor': 5,
    #             'transport_fixed_factor': 1,
    #             'action_space': 4}
    #
    # run_simulation('GA', settings)

    settings = {'transport_distance_factor': 5,
                'transport_fixed_factor': 1,
                'action_space': 8}
    run_simulation('PPO', settings)
