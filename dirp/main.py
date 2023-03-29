import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from dirp.AI4LEnvironment import AI4LEnvironment
from dirp.PPO_env import PPO_env
from dirp.PPO_util import linear_schedule
from dirp.genetic_algorithm import GeneticAlgorithm


def run_simulation(policy: str) -> None:
    # create dataframe for saving results
    df = pd.DataFrame(columns=['iteration', 'inventory', 'action', 'reward', 'avgCost', 'HoldCost', 'LostCost', 'RoutingCost'])

    # create the environment
    env = AI4LEnvironment()

    # reset the environment
    obs = env.reset()
    iteration = 0
    done = False

    while done == False:
        print('\n', iteration, '------------------------------------')

        # SS policy
        if policy == 'SS':
            action = np.less_equal(obs, env.s).astype(int)
            action[0] = 1

        # Genetic Algorithm policy
        if policy == 'GA':
            gen = GeneticAlgorithm(env)
            action = gen.run()

        # PPO policy
        if policy == 'PPO':
            model = None
            if iteration == 0:
                env_PPO = PPO_env()
                # env = make_vec_env('PPO-v0', n_envs=4)
                model = PPO('MlpPolicy', env_PPO, gamma=0.95, learning_rate=linear_schedule(0.001), verbose=0)
                model.learn(total_timesteps=60000)
                model.save("ppo_dirp")
            action, _states = model.predict(obs, deterministic=False)

        # copy inventory before demand and action are taken
        obs_old = obs.copy()

        # take the action
        obs, reward, done, info = env.step(action)
        print('inventory: ', obs)
        print('reward: ', reward)
        print('average cost: ', env.avgCost)

        # save results
        df = df.append({'iteration': iteration, 'inventory': obs_old, 'action': action, 'reward': reward, 'avgCost': env.avgCost, 'HoldCost': info['HoldingCost'], 'LostCost': info['LostCost'], 'RoutingCost': info['TransportationCost']}, ignore_index=True)

        iteration += 1

    df.to_csv(f'results_{policy}.csv', index=False)


if __name__ == '__main__':
    run_simulation('SS')
    run_simulation('GA')
    run_simulation('PPO')
