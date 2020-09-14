import os
import sys
import argparse
import warnings
import numpy as np
import pickle as pkl
from pathlib import Path

import gym
import minerl

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

    parser.add_argument('-conv-vec', '--convert-vec', help='Convert npz vector data into pkl', action='store_true')
    parser.add_argument('-conv-full', '--convert-full', help='Convert pov and vector npz data into pkl', action='store_true')
    parser.add_argument('--view-vec', help='View vector data as npz or pkl', action='store_true')
    parser.add_argument('--view-full', help='View pov and vector data as npz or pkl', action='store_true')

    parser.add_argument('-flatten', '--flatten-states', help='Convert current_state and next_state vector to a 12352-length array', action='store_true')
    parser.add_argument('-aggregate', '--aggregate-states', help='Add current_state and next_state vector to pov as a fourth channel', action='store_true')
    parser.add_argument('--traj-use', help='Number of trajectories used to create expert', type=int, default=5, choices=range(1, 211))

    parser.add_argument('--view-pkl', help='View pkl file', action='store_true')
    parser.add_argument('--view-npy', help='View npy file (only MineRL envs)', action='store_true')
    parser.add_argument('--view-npz', help='View npz file (only MineRL envs)', action='store_true')
    parser.add_argument('--view-npz-final', help='View npz file (only MineRL envs)', action='store_true')

    parser.add_argument('--episodic', help='Episodic data ', action='store_true')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    args = parser.parse_args()
    return args

def MineRL_flatten_state(state_unflattened):
    n, state = len(state_unflattened), np.array([])
    for i in range(n):
        state = np.append(state, state_unflattened[i].flatten()) # for competition envs
    return state

def MineRL_aggregate_state(state):
    vector_scale = 1 / 255
    pov = state['pov']
    vector_scaled = state['vector'] / vector_scale
    num_elem = pov.shape[-3] * pov.shape[-2]
    vector_channel = np.tile(vector_scaled, num_elem // vector_scaled.shape[-1]).reshape(*pov.shape[:-1], -1)  # noqa
    return np.concatenate([pov, vector_channel], axis=-1)

def main():
    args = get_args()
    env_id = args.env
    # algo = args.algo
    # exp_id = args.exp_id

    # Convert vector component of expert data into pkl format
    if ((args.convert_vec) and ('MineRL' in env_id)): # If MineRL env, combine npz-s and convert to pkl
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
                        expert_data_pkl[start+step].insert(4, False)
            expert_data_pkl[total_trajectory_length-1][4] = True
            trajectory_count += 1
            print('file count: ', trajectory_count)

            if trajectory_count == args.traj_use:
                print('Used {} trajectories to create expert data. Exiting'.format(trajectory_count))
                break
        # print(trajectory_count) # sanity check if all files were parsed

        print('Total number of steps in expert: ', total_trajectory_length)
        np.savez(os.path.join(path_pkl, env_id[:-3].lower()+'_vector_'+str(trajectory_count)), **expert_data_npz)
        with open(os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_vector_'+str(trajectory_count))), 'wb') as handle:
            pkl.dump(expert_data_pkl, handle, protocol=pkl.HIGHEST_PROTOCOL)


    # Convert complete expert data into pkl format
    if ((args.convert_full) and ('MineRL' in env_id)):
        expert_data_npz, expert_data_pkl, trajectory_count = {'reward': [], 'observation$vector': [], 'action$vector': []}, [], 0
        data = minerl.data.make(env_id, data_dir=path_npz)

        for path in Path(os.path.join(path_npz, env_id)).glob('*'):
            experience = data.load_data(str(path))

            for items in experience:
                # import pdb; pdb.set_trace()
                # print(items)

                if args.flatten_states: # convert current_state and next_state to a 12352-length array
                    current_state = MineRL_flatten_state([items[0]['pov'], items[0]['vector']])
                    next_state = MineRL_flatten_state([items[3]['pov'], items[3]['vector']])
                elif args.aggregate_states: # add current_state and next_state vector to pov as a fourth channel
                    current_state = MineRL_aggregate_state(items[0])
                    next_state = MineRL_aggregate_state(items[3])

                expert_data_npz['reward'].append(items[2])
                expert_data_npz['observation$vector'].append(current_state)
                expert_data_npz['action$vector'].append(items[1]['vector'])
                expert_data_pkl.append([current_state, items[1]['vector'], items[2], next_state, items[4]])

            trajectory_count += 1
            print('file count: ', trajectory_count)

            if trajectory_count == args.traj_use:
                print('Used {} trajectories to create expert data. Exiting'.format(trajectory_count))
                break
        # print(trajectory_count) # sanity check if all files were parsed

        if args.flatten_states:
            save_path_npz = os.path.join(path_pkl, env_id[:-3].lower()+'_flat_'+str(trajectory_count))
            save_path_pkl = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_flat_'+str(trajectory_count)))
        elif args.aggregate_states:
            save_path_npz = os.path.join(path_pkl, env_id[:-3].lower()+'_conv_'+str(trajectory_count))
            save_path_pkl = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_conv_'+str(trajectory_count)))
        else:
            save_path_npz = os.path.join(path_pkl, env_id[:-3].lower()+'_'+str(trajectory_count))
            save_path_pkl = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_'+str(trajectory_count)))

        np.savez(save_path_npz, **expert_data_npz)
        with open(save_path_pkl, 'wb') as handle:
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
            if args.view_vec:
                path_expert = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_vector_'+str(args.traj_use)))
            elif args.view_full:
                if args.flatten_states:
                    path_expert = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_flat_'+str(args.traj_use)))
                elif args.aggregate_states:
                    path_expert = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_conv_'+str(args.traj_use)))
                else:
                    path_expert = os.path.join(path_pkl, '{}.pkl'.format(env_id[:-3].lower()+'_'+str(args.traj_use)))
            with open(path_expert, 'rb') as handle:
                expert_data = pkl.load(handle)

        trajectory_length = len(expert_data)
        for index in range(trajectory_length):
            import pdb; pdb.set_trace()
            print('Step {} of the expert data: '.format(index), expert_data[index])
            print(expert_data[index])
            for i in range(5):
                print(type(expert_data[index][i]))
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


    if args.view_npz: # view original vector npz
        print('Here are some stats of the MineRL expert... ')
        for path in Path(os.path.join(path_npz, env_id)).rglob('*.npz'):
            print(path)
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
        if args.view_vec:
            expert_data = np.load(os.path.join(path_pkl, env_id[:-3].lower()+'_vector_'+str(args.traj_use)+'.npz'), allow_pickle=True) #Gym-envs
        elif args.view_full:
            if args.flatten_states:
                path_expert = os.path.join(path_pkl, env_id[:-3].lower()+'_flat_'+str(args.traj_use)+'.npz')
            elif args.aggregate_states:
                path_expert = os.path.join(path_pkl, env_id[:-3].lower()+'_conv_'+str(args.traj_use)+'.npz')
            else:
                path_expert = os.path.join(path_pkl, env_id[:-3].lower()+'_'+str(args.traj_use)+'.npz')
        expert_data = np.load(path_expert, allow_pickle=True) #Gym-envs

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