"""Tests for alpacka.trainers.td."""

import numpy as np

from alpacka.data import Episode, Transition
from alpacka.trainers import td


def test_target_n_return():
    """A rudimentary test of target_n_return"""

    transition_batch = Transition(observation=np.expand_dims(np.arange(10), 1),
                                  action='not used',
                                  reward=np.arange(10),
                                  done=[False] * 10,
                                  next_observation=np.expand_dims(
                                      np.arange(1, 11), 1),
                                  agent_info='not used')

    e = Episode(transition_batch=transition_batch,
                return_='not used', solved='not used', truncated=False)

    datapoints = td.target_n_return(e, 1, 0.99)

    bootstrap_gamma = np.full((10, 1), 0.99)
    bootstrap_gamma[9, 0] = 0.0
    bootstrap_obs = np.expand_dims(np.arange(1, 11), 1)
    cum_reward = np.expand_dims(np.arange(10), 1)

    assert np.array_equal(datapoints.bootstrap_gamma, bootstrap_gamma), \
            'bootstrap_gamma error'
    assert np.array_equal(datapoints.bootstrap_obs, bootstrap_obs), \
            'bootstrap_obs error'
    assert np.array_equal(datapoints.cum_reward, cum_reward), \
            'cum_reward error'
