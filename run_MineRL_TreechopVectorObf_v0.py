# -*- coding: utf-8 -*-
"""Train or test algorithms on MineRLTreechopVectorObf-v0.

- Author: Prabhasa Kalkur
- Contact: prabhasa.94@gmail.com

Config file for algo: --cfg-path
Expert demo for fD algos: --demo-path (system-dependent)
Env name: env_name
WANDB logs: wandb.init (system-dependent)
"""

import argparse
import datetime
import minerl
import os

import gym
import envs
import numpy as np
from envs.wrappers import wrap
from rl_algorithms import build_agent
import rl_algorithms.common.env.utils as env_utils
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import Config

from xvfbwrapper import Xvfb # only for ecelbw00202

import wandb
# wandb.config["more"] = "custom"

def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/MineRLTreechopVectorObf_v0/dqfd.py", # PARAM 1: ALGORITHM
        help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--off-render", dest="render", action="store_true", help="turn off rendering"
    )
    parser.add_argument(
        "--render-after",
        type=int,
        default=0,
        help="start rendering after the input number of episode",
    )
    parser.add_argument(
        "--log", dest="log", action="store_false", help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=10, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=50, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=8000, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=3,
        help="number of test during training",
    )
    parser.add_argument(
        "--demo-path",
        type=str,
        # PARAM 2: FOR FD ALGOS
        default="./data/minerltreechopvectorobf_flat_5.pkl",
        help="demonstration path for learning from demo",
    )
    parser.add_argument(
        "--integration-test",
        dest="integration_test",
        action="store_true",
        help="indicate integration test",
    )

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # PARAM 3: INITILAIZE WANDB
    wandb.init(name='dqn_mtc_obf_5', project="lensminerl_treechop_obf", dir='/home/grads/p/prabhasa/MineRL2020/medipixel', group='september', reinit=True, sync_tensorboard=True) # ecelbw00202
    # wandb.init(name='dqn_mtc_obf_1', project="wandb_on_minerl", dir='C:/GitHub/MineRL-NeurIPS-2020', group='dry_run', reinit=True, sync_tensorboard=True) # PK laptop: locally cloned repo
    # wandb.init(name='dqn_mtc_obf_1', project="wandb_on_minerl", dir='C:/MineRL/medipixel', group='dry_run', reinit=True, sync_tensorboard=True) # PK laptop: locally run code
    # wandb.tensorboard.patch(tensorboardX=True, pytorch=True)

    # PARAM 3: env initialization and wrappers
    env_name = "MineRLTreechopVectorObf-v0"
    # env_name = "MineRLObtainDiamondVectorObf-v0"
    env = gym.make(env_name)
    env = wrap(env, conv=False, discrete=True, seed=args.seed) # data_dir=None as MINERL_DATA_ROOT has been set
    env = env_utils.set_env(env, args)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)

    # If running integration test, simplify experiment
    if args.integration_test:
        cfg = common_utils.set_cfg_for_intergration_test(cfg)

    # PK: Added np.array to obs_space and changed is_discrete
    cfg.agent.env_info = dict(
        name=env_name,
        observation_space=np.array(env.observation_space),
        action_space=env.action_space,
        is_discrete=True,  # PARAM 4.2: FOR DISCRETE ACTION SPACES
    )
    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)

    if not args.test:
        agent.train()
    else:
        import pdb; pdb.set_trace()
        agent.test()

    wandb.join()


if __name__ == "__main__":
#     vdisplay = Xvfb(width=1280, height=740, colordepth=16)
#     vdisplay.start()
    main()
#     vdisplay.stop()
