import gym
import numpy as np

class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination, use_tuple=False):
        if use_tuple:
            assert isinstance(env.observation_space[0], gym.spaces.Box)
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination
        self.use_tuple = use_tuple

        if self.use_tuple:
            low = self.observation(
                tuple([space.low for space in self.observation_space]))
            high = self.observation(
                tuple([space.high for space in self.observation_space]))
            dtype = self.observation_space[0].dtype
            pov_space = gym.spaces.Box(low=low[0], high=high[0], dtype=dtype)
            inventory_space = self.observation_space[1]
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inventory_space))
        else:
            low = self.observation(self.observation_space.low)
            high = self.observation(self.observation_space.high)
            dtype = self.observation_space.dtype
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=dtype)

    def observation(self, observation):
        if self.use_tuple:
            new_observation = list(observation)
            new_observation[0] = np.moveaxis(
                observation[0], self.source, self.destination)
            return tuple(new_observation)
        else:
            return np.moveaxis(observation, self.source, self.destination)