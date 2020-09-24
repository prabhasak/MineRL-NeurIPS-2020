import gym
import numpy as np
from skimage import img_as_float


class MineRLObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # # Added by PK: only for basic envs
        # self.list_observation_space = []
        # for key in self.observation_space.spaces:
        #     self.list_observation_space.append(dict(self.observation_space.spaces)[key])
        # self.observation_space = np.array(gym.spaces.Tuple((self.list_observation_space)))

        # Original code
        # self.vector_scale = 1/255
        # self.pov_scale = 1 / 255
        self.observation_space = gym.spaces.Tuple((self.observation_space['pov'], self.observation_space['vector']))

        # # Method 2: to avoid changing agent.py & learner.py - incorrect syntax (and impossible)
        # self.observation_space = gym.spaces.Tuple(((np.array(self.observation_space['pov']).flatten())[0], (np.array(self.observation_space['vector']).flatten())[0]))


    def observation(self, observation):

        # # Added by PK: only for basic envs
        # return (self.list_observation_space)

        # # Original code
        # return (observation['pov']//self.pov_scale, observation['vector'])
        return (img_as_float(observation['pov']), observation['vector'])
        # # Method 2: to avoid changing agent.py & learner.py - incorrect syntax (and impossible)
        # return (np.array(self.observation_space['pov']).flatten()/255)[0], (np.array(self.observation_space['vector']).flatten())[0]

class MineRLPOVWithVectorWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+vector (or K+vector for gray-scaled image) channels.
    """
    def __init__(self, env):
        super().__init__(env)

        self._vector_scale = 1 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later

        pov_space = self.env.observation_space.spaces['pov']
        vector_space = self.env.observation_space.spaces['vector']

        low = self.observation({'pov': pov_space.low, 'vector': vector_space.low})
        high = self.observation({'pov': pov_space.high, 'vector': vector_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        # pov = observation['pov']
        # vector_scaled = observation['vector'] / self._vector_scale
        pov_scaled = img_as_float(observation['pov'])
        vector = observation['vector']
        num_elem = pov_scaled.shape[-3] * pov_scaled.shape[-2]
        vector_channel = np.tile(vector, num_elem // vector.shape[-1]).reshape(*pov_scaled.shape[:-1], -1)  # noqa
        return np.concatenate([pov_scaled, vector_channel], axis=-1)
