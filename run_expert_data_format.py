import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from collections import OrderedDict
import rl_algorithms.common.helper_functions as common_utils

path_pkl = 'C:/MineRL/rl_algorithms/data/'
path_npy = 'C:/MineRL/envs/32-means/'
path_npz = 'C:/MineRL/envs/'

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

algo_list = ['a2c', 'bcsac', 'bcddpg', 'dqn', 'ddpg', 'ddpgfd', 'dqn', 'dqfd', 'sac', 'sacfd', 'ppo', 'td3']

def env_list():
    env_openai = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2',
                'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0']
    env_minerl_basic = ['MineRLTreechop-v0', 'MineRLNavigate-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateDense-v0', 'MineRLObtainDiamond-v0',
                        'MineRLObtainDiamondDense-v0', 'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0']
    env_minerl_competition = ['MineRLTreechopVectorObf-v0', 'MineRLNavigateVectorObf-v0', 'MineRLNavigateExtremeDense-v0', 'MineRLNavigateDenseVectorObf-v0',
                            'MineRLNavigateExtremeDenseVectorObf-v0', 'MineRLObtainDiamondVectorObf-v0', 'MineRLObtainDiamondDenseVectorObf-v0',
                            'MineRLObtainIronPickaxeVectorObf-v0', 'MineRLObtainIronPickaxeDenseVectorObf-v0']
    return (env_openai + env_minerl_basic + env_minerl_competition)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='MineRLTreechopVectorObf-v0', choices=env_list())
    parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False, choices=algo_list)
    parser.add_argument('--convert', help='Convert data', action='store_true')
    # parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=0, type=int)
    # parser.add_argument('--optimal', help='Whether model is optimal or suboptimal', action='store_true')
    parser.add_argument('--view-pkl', help='View pkl file', action='store_true')
    parser.add_argument('--view-npy', help='View npy file (only MineRL envs)', action='store_true')
    parser.add_argument('--view-npz', help='View npz file (only MineRL envs)', action='store_true')
    parser.add_argument('--view-npz-final', help='View npz file (only MineRL envs)', action='store_true')
    parser.add_argument('--episodic', help='Episodic data ', action='store_true')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    env_id = args.env
    algo = args.algo
    # exp_id = args.exp_id

    # common_utils.set_random_seed(args.seed, env)

    if ((args.convert) and ('MineRL' in env_id)): # If MineRL env, combine npz-s and convert to pkl
        expert_data_pkl = OrderedDict([('reward', []), ('observation$vector', []), ('action$vector', [])])
        expert_data_npz = {'reward': [], 'observation$vector': [], 'action$vector': []}
        file_count = 0

        # Step 1: Accessing the npz files recursively and appending them
        for path in Path(os.path.join(path_npz, env_id)).rglob('*.npz'):
            # print(path)
            expert_trajectory = np.load(path, allow_pickle=True) #MineRL-envs
            for key in expert_data_pkl:
                expert_data_pkl[key].append(expert_trajectory[key])
                expert_data_npz[key].append(expert_trajectory[key])
                # print(len(expert_data_pkl[key][file_count])) # shape of each trajectory
            file_count += 1
        
        print(file_count) # sanity check if all files were parsed
        np.savez(os.path.join(path_pkl, env_id[:-3].lower()), **expert_data_npz)

        # Step 2: Converting to .pkl format
        with open(os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower())), 'wb') as handle:
            pkl.dump(expert_data_pkl, handle, protocol=pkl.HIGHEST_PROTOCOL)


    if args.view_pkl: # Step 3: view combined pkl
        print('Here are some stats of the MineRL expert... ')
        if 'MineRL' in env_id:
            with open(os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower())), 'rb') as handle: #MineRL-envs
                expert_data = pkl.load(handle)
        else:
            if 'lunarlander' in env_id.lower():
                if 'continuous' in env_id.lower():
                    env_id = 'lunarlander_continuous-v2'
                else:
                    env_id = 'lunarlander_discrete-v2'
            with open(os.path.join(path_pkl, '{}_demo.pkl'.format(env_id[:-3].lower())), 'rb') as handle: #Gym-envs
                expert_data = pkl.load(handle)
        
        if 'MineRL' in env_id:
            trajectory_max = {'reward': [], 'observation$vector': [], 'action$vector': []}
            trajectory_min = {'reward': [], 'observation$vector': [], 'action$vector': []}
            trajectory_shape = {'reward': [], 'observation$vector': [], 'action$vector': []}
            for keys in expert_data:
                print(keys)
                for trajectory, trajectory_key in enumerate(expert_data[keys]):
                    import pdb; pdb.set_trace()
                    trajectory_max[keys].append(np.amax(trajectory_key.astype(int), axis=0))
                    trajectory_min[keys].append(np.amin(trajectory_key.astype(int), axis=0))
                    trajectory_shape[keys].append(len(expert_data[keys]))
                    if args.episodic:
                        import pdb; pdb.set_trace()
                        print(trajectory_max[:][trajectory], trajectory_min[:][trajectory], trajectory_shape[:][trajectory])
            print(trajectory_max)
            print(trajectory_min)
            print(trajectory_shape)
        else:
            import pdb; pdb.set_trace()
            for keys in expert_data:
                print(keys)
                # print(expert_data[keys])
                print(np.amax(expert_data[keys], axis=0)) # otherwise will get a 64-length numpy array
                print(np.amin(expert_data[keys], axis=0)) # otherwise will get a 64-length numpy array
                print(expert_data[keys].shape)



    if args.view_npy: # Step 4: view npy
        print('Here are some stats of the MineRL expert... ')
        expert_data = np.load(os.path.join(path_npy, env_id+'.npy'), allow_pickle=True) #Gym-envs

        trajectory_max = {'reward': [], 'observation$vector': [], 'action$vector': []}
        trajectory_min = {'reward': [], 'observation$vector': [], 'action$vector': []}
        trajectory_shape = {'reward': [], 'observation$vector': [], 'action$vector': []}
        import pdb; pdb.set_trace()
        print(expert_data.shape)
        for keys in expert_data:
            print(keys)
            for trajectory, trajectory_key in enumerate(expert_data[keys]):
                import pdb; pdb.set_trace()
                trajectory_max[keys].append(np.amax(trajectory_key.astype(int), axis=0))
                trajectory_min[keys].append(np.amin(trajectory_key.astype(int), axis=0))
                trajectory_shape[keys].append(len(expert_data[keys]))
                if args.episodic:
                    import pdb; pdb.set_trace()
                    print(trajectory_max[:][trajectory],trajectory_min[:][trajectory], trajectory_shape[:][trajectory])
        print(trajectory_max)
        print(trajectory_min)
        print(trajectory_shape)


    if args.view_npz: # view individual npz
        print('Here are some stats of the MineRL expert... ')
        for path in Path(os.path.join(path_npz, env_id)).rglob('*.npz'):
            # print(path)
            import pdb; pdb.set_trace()
            expert_trajectory = np.load(path, allow_pickle=True) #MineRL-envs
            for keys in expert_trajectory:
                print(keys)
                # print(expert_trajectory[keys])
                print(np.amax(np.amax(expert_trajectory[keys], axis=0))) # otherwise will get a 64-length numpy array
                print(np.amin(np.amin(expert_trajectory[keys], axis=0))) # otherwise will get a 64-length numpy array
                print(expert_trajectory[keys].shape)

    if args.view_npz_final: # Step 5: view combined npz
        print('Here are some stats of the MineRL expert... ')
        expert_data = np.load(os.path.join(path_pkl, env_id[:-3].lower()+'.npz'), allow_pickle=True) #Gym-envs

        trajectory_max = {'reward': [], 'observation$vector': [], 'action$vector': []}
        trajectory_min = {'reward': [], 'observation$vector': [], 'action$vector': []}
        trajectory_shape = {'reward': [], 'observation$vector': [], 'action$vector': []}
        for keys in expert_data:
            print(keys)
            for trajectory, trajectory_key in enumerate(expert_data[keys]):
                import pdb; pdb.set_trace()
                trajectory_max[keys].append(np.amax(trajectory_key.astype(int), axis=0))
                trajectory_min[keys].append(np.amin(trajectory_key.astype(int), axis=0))
                trajectory_shape[keys].append(len(expert_data[keys]))
                if args.episodic:
                    import pdb; pdb.set_trace()
                    print(trajectory_max[:][trajectory], trajectory_min[:][trajectory], trajectory_shape[:][trajectory])
        print(trajectory_max)
        print(trajectory_min)
        print(trajectory_shape)

if __name__ == '__main__':
    main()