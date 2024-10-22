import warnings
import argparse
import os

import minerl
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module='gym')

def env_choices():
    env_minerl_basic = ['MineRLTreechop-v0', 'MineRLNavigate-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateDense-v0', 'MineRLObtainDiamond-v0',
                        'MineRLObtainDiamondDense-v0', 'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0']
    env_minerl_competition = ['MineRLTreechopVectorObf-v0', 'MineRLNavigateVectorObf-v0', 'MineRLNavigateExtremeDense-v0', 'MineRLNavigateDenseVectorObf-v0',
                            'MineRLNavigateExtremeDenseVectorObf-v0', 'MineRLObtainDiamondVectorObf-v0', 'MineRLObtainDiamondDenseVectorObf-v0',
                            'MineRLObtainIronPickaxeVectorObf-v0', 'MineRLObtainIronPickaxeDenseVectorObf-v0']
    return (env_minerl_basic + env_minerl_competition)

parser = argparse.ArgumentParser()

parser.add_argument('--env', default=None, choices=env_choices())
parser.add_argument('--num-actions', type=int, default=32)
# parser.add_argument('--data-dir', default='C:/MineRL/envs/') # PK's Windows laptop
parser.add_argument('--data-dir', default=os.getenv('MINERL_DATA_ROOT', 'data'))


def main():
    args = parser.parse_args()
    if args.env is None:
        env_list = []
        for env_name in os.listdir(args.data_dir):
            if 'VectorObf' in env_name:
                env_list.append(env_name)
    else:
        env_list = [args.env]

    for env_name in env_list:
        print(f'Generating {args.num_actions}-means for {env_name}')
        data = minerl.data.make(env_name)
        actions = []
        for trajectory_name in tqdm(list(data.get_trajectory_names())):
            try:
                for _, action, _, _, _ in data.load_data(trajectory_name):
                    actions.append(action['vector'])
            except TypeError:
                pass
        actions = np.stack(actions)
        print('computing k-means...')
        kmeans = KMeans(n_clusters=args.num_actions, verbose=1, random_state=0).fit(actions)
        print(kmeans)
        file_dir = os.path.join(args.data_dir, f'{args.num_actions}-means')
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file = os.path.join(file_dir, env_name + '.npy')
        np.save(file, kmeans.cluster_centers_)

if __name__ == '__main__':
    main()