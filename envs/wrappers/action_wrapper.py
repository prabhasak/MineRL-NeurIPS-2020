import gym
import numpy as np


class MineRLActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # # Added by PK: only for basic envs
        # self.list_action_space = []
        # for key in self.action_space.spaces:
        #     self.list_action_space.append(dict(self.action_space.spaces)[key])
        # self.action_space = np.array(gym.spaces.Tuple((self.list_action_space)))

        # Original code
        self.action_space = (self.action_space['vector'])

    def action(self, action):
        return dict(vector=action)

    def reverse_action(self, action):
        # # Added by PK: only for basic envs
        # return (self.list_action_space)
        return action['vector']
