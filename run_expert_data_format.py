import os
import sys
import argparse
import warnings
import numpy as np
import pickle as pkl
from pathlib import Path

#Windows
# path_pkl = 'C:/MineRL/medipixel/data/'
# path_npy = 'C:/MineRL/data/32-means/'
# path_npz = 'C:/MineRL/data/'

#ecelbw002
path_pkl = './data/'
path_npy = './data/32-means/'
path_npz = '/home/grads/p/prabhasa/MineRL2020/data'

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
    parser.add_argument('--convert', help='Convert npz data into pkl', action='store_true')
    parser.add_argument('--traj-use', help='Number of trajectories used to create expert', type=int, default=210, choices=range(1, 210))
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
    # algo = args.algo
    # exp_id = args.exp_id

    # common_utils.set_random_seed(args.seed, env)

    if ((args.convert) and ('MineRL' in env_id)): # If MineRL env, combine npz-s and convert to pkl
        # expert_data_pkl = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        expert_data_npz = {'reward': [], 'observation$vector': [], 'action$vector': []}

        expert_data_pkl, total_trajectory_length, trajectory_count = [], 0, 0
        for path in Path(os.path.join(path_npz, env_id)).rglob('*.npz'):
            expert_trajectory = np.load(path, allow_pickle=True) #MineRL-envs

            current_trajectory_length = expert_trajectory['action$vector'].shape[0]
            total_trajectory_length += current_trajectory_length
            print('Current trajectory length: ', current_trajectory_length)
            start = total_trajectory_length-current_trajectory_length

            for key in expert_trajectory:
                expert_data_npz[key].append(expert_trajectory[key])

                if key == 'reward':
                    for step in range(current_trajectory_length):
                        expert_data_pkl.append([expert_trajectory[key][step]])
                if key == 'observation$vector':
                    for step in range(current_trajectory_length):
                        expert_data_pkl[start+step].insert(0, expert_trajectory[key][step])
                        expert_data_pkl[start+step].insert(3, expert_trajectory[key][step+1])
                if key == 'action$vector':
                    for step in range(current_trajectory_length):
                        expert_data_pkl[start+step].insert(1, expert_trajectory[key][step])
                        expert_data_pkl[start+step].insert(4, 'False')
            expert_data_pkl[total_trajectory_length-1][4] = 'True'
            trajectory_count += 1
            print('file count: ', trajectory_count)

            if trajectory_count == args.traj_use:
                print('Used {} trajectories to create expert data. Exiting'.format(trajectory_count))
                break

        # print(trajectory_count) # sanity check if all files were parsed
        print('Total number of steps in expert: ', total_trajectory_length)
        np.savez(os.path.join(path_pkl, env_id[:-3].lower()+'_'+trajectory_count), **expert_data_npz)

        # Step 2: Converting to .pkl format
        with open(os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_'+trajectory_count)), 'wb') as handle:
            pkl.dump(expert_data_pkl, handle, protocol=pkl.HIGHEST_PROTOCOL)


    if args.view_pkl: # Step 3: view combined pkl
        print('Here are some stats of the {} expert... '.format(env_id))
        if 'lunarlander' in env_id.lower():
            if 'continuous' in env_id.lower():
                env_id = 'lunarlander_continuous-v2'
            else:
                env_id = 'lunarlander_discrete-v2'
            with open(os.path.join(path_pkl, '{}_demo.pkl'.format(env_id[:-3].lower())), 'rb') as handle:
                expert_data = pkl.load(handle)
        elif 'MineRL' in env_id:
            with open(os.path.join(path_pkl, '{}_demo.pkl'.format(env_id[:-3].lower()+'_'+args.traj_use)), 'rb') as handle:
                expert_data = pkl.load(handle)

        trajectory_length = len(expert_data)
        for index in range(trajectory_length):
            import pdb; pdb.set_trace()
            print('Step {} of the expert data: '.format(index), expert_data[index])
            # if args.episodic and expert_data[-1]!='True':


    if args.view_npy: # view npy
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
                print(expert_trajectory[keys])
                print(np.amax(np.amax(expert_trajectory[keys], axis=0))) # otherwise will get a 64-length numpy array
                print(np.amin(np.amin(expert_trajectory[keys], axis=0))) # otherwise will get a 64-length numpy array
                print(expert_trajectory[keys].shape)


    if args.view_npz_final: # view combined npz
        print('Here are some stats of the MineRL expert... ')
        expert_data = np.load(os.path.join(path_pkl, env_id[:-3].lower()+'_'+args.traj_use+'.npz'), allow_pickle=True) #Gym-envs

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