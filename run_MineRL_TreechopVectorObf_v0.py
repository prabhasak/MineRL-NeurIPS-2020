# -*- coding: utf-8 -*-
"""Train or test algorithms on MineRLTreechopVectorObf-v1.

- Authors: Prabhasa Kalkur, Kishan P B 
- Contact: prabhasa.94@gmail.com

Config file for algo: --cfg-path (algo-dependent)
Pretrain or test model: --load-from (run-dependent)
Expert demo for fD algos: --demo-path (trajectory-dependent)
WANDB logs: wandb.init (system-dependent)
Env name: env_name (env-dependent)
"""

import argparse
import datetime
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/MineRLTreechopVectorObf_v0/dqfd.py",
        help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        # default="./checkpoint/MineRLTreechopVectorObf-v0/DQfDAgent/200915_024742/571f207_ep_50.pt",
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--off-render", dest="render", action="store_false", help="turn off rendering"
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
        "--save-period", type=int, default=20, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=100, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=8000, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=4,
        help="number of test during training",
    )
    parser.add_argument(
        "--demo-path",
        type=str,
        # default=None,
        default = "./data/minerltreechopvectorobf_flat_5.pkl",
        help="demonstration path for learning from demo",
    )
    parser.add_argument(
        "--integration-test",
        dest="integration_test",
        action="store_true",
        help="indicate integration test",
    )
    parser.add_argument(
        "--env", default="MineRLTreechopVectorObf-v0", help="env for filename purposes"
    )
    parser.add_argument(
        "--algo", default="DQfD", help="algo for filename purposes"
    )
    parser.add_argument(
        "-conv", "--conv-layer", action="store_true", help="if conv layer used"
    )

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()
    filename = '-conv' if args.conv_layer else '-flat'

    # INITIALIZE WANDB
    if args.demo_path is not None:
        if args.demo_path[-6] == '_':
            wandb.init(name=args.algo+'-'+str(args.demo_path[-5])+filename, project="lensminerl", dir='/home/grads/p/prabhasa/MineRL2020/medipixel', group='mtc_obf_sep', reinit=True, sync_tensorboard=True) # ecelbw00202
        else:
            wandb.init(name=args.algo+'-'+str(args.demo_path[-6:-4])+filename, project="lensminerl", dir='/home/grads/p/prabhasa/MineRL2020/medipixel', group='mtc_obf_sep', reinit=True, sync_tensorboard=True) # ecelbw00202
    else:
        wandb.init(name=args.algo+filename, project="lensminerl", dir='/home/grads/p/prabhasa/MineRL2020/medipixel', group='mtc_obf_sep', reinit=True, sync_tensorboard=True) # ecelbw00202
    # wandb.init(name='Rainbow-DQN-flat', project="minerlpk", dir='C:/MineRL/medipixel', group='dry_run', reinit=True, sync_tensorboard=True) # PK laptop: locally run code
    # wandb.tensorboard.patch(tensorboardX=True, pytorch=True)

    # INITIALIZE ENV
    env_name = args.env
    env = gym.make(env_name)
    env = wrap(env, conv=args.conv_layer, discrete=True, seed=args.seed) # data_dir=None as MINERL_DATA_ROOT has been set
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

    cfg.agent.env_info = dict(
        name=env_name,
        observation_space=np.array(env.observation_space), # PK: FOR DISCRETE ACTION SPACES
        action_space=env.action_space,
        is_discrete=True,  # PK: FOR DISCRETE ACTION SPACES
        conv_layer=args.conv_layer, # PK: IF CONV LAYER USED
    )
    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)

    if not args.test:
        agent.train()
    else:
        agent.test()

    wandb.join()


if __name__ == "__main__":
    main()