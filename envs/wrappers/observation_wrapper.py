import gym
import numpy as np


class MineRLObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # # Added by PK: only for basic envs
        # self.list_observation_space = []
        # for key in self.observation_space.spaces:
        #     self.list_observation_space.append(dict(self.observation_space.spaces)[key])
        # self.observation_space = np.array(gym.spaces.Tuple((self.list_observation_space)))

        # Original code
        self.observation_space = gym.spaces.Tuple((self.observation_space['pov'], self.observation_space['vector']))

        # # Method 2: to avoid changing agent.py & learner.py - incorrect syntax (and impossible)
        # self.observation_space = gym.spaces.Tuple(((np.array(self.observation_space['pov']).flatten())[0], (np.array(self.observation_space['vector']).flatten())[0]))


    def observation(self, observation):

        # # Added by PK: only for basic envs
        # return (self.list_observation_space)

        # # Original code
        return (observation['pov'], observation['vector'])

        # # Method 2: to avoid changing agent.py & learner.py - incorrect syntax (and impossible)
        # return (np.array(self.observation_space['pov']).flatten()/255)[0], (np.array(self.observation_space['vector']).flatten())[0]
