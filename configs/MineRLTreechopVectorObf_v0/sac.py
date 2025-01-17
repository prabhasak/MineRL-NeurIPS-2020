"""Config for SAC on MineRLTreechop-v0.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from rl_algorithms.common.helper_functions import identity

agent = dict(
    type="SACAgent",
    hyper_params=dict(
        gamma=0.995,
        tau=5e-3,
        buffer_size=int(1e5),
        batch_size=128,
        initial_random_action=int(5e3),
        multiple_update=1,  # multiple learning updates
        policy_update_freq=2,
        w_entropy=1e-3,
        w_mean_reg=0.0,
        w_std_reg=0.0,
        w_pre_activation_reg=0.0,
        auto_entropy_tuning=True,
    ),
    learner_cfg=dict(
        type="SACLearner",
        backbone=dict(actor=dict(), critic_vf=dict(), critic_qf=dict()),
        head=dict(
            actor=dict(
                type="TanhGaussianDistParams",
                configs=dict(hidden_sizes=[256, 256], output_activation=identity,),
            ),
            critic_vf=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_activation=identity, output_size=1,
                ),
            ),
            critic_qf=dict(
                type="MLP",
                configs=dict(
                    hidden_sizes=[256, 256], output_activation=identity, output_size=1,
                ),
            ),
        ),
        optim_cfg=dict(
            lr_actor=3e-4,
            lr_vf=3e-4,
            lr_qf1=3e-4,
            lr_qf2=3e-4,
            lr_entropy=3e-4,
            weight_decay=0.0,
        ),
    ),
)
