"""Temporal difference trainer."""

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers


@gin.configurable
def target_n_return(episode, n, gamma):
    """Calculates targets with n-step bootstraped returns."""
    assert n >= 1, 'TD(0) does not make sense'
    truncated = episode.truncated
    rewards = episode.transition_batch.reward
    ep_len = len(episode.transition_batch.observation)
    cum_reward = 0

    cum_rewards = []
    bootstrap_obs = []
    bootstrap_gammas = []
    for curr_obs_index in reversed(range(ep_len)):
        reward_to_remove = 0 if curr_obs_index + n >= ep_len \
            else gamma ** (n - 1) * rewards[curr_obs_index + n]
        cum_reward -= reward_to_remove
        cum_reward *= gamma
        cum_reward += rewards[curr_obs_index]

        bootstrap_index = min(curr_obs_index + n, ep_len)
        bootstrap_ob = \
            episode.transition_batch.next_observation[bootstrap_index - 1]
        bootstrap_lambda = gamma ** (bootstrap_index - curr_obs_index)
        if bootstrap_index == ep_len and not truncated:
            bootstrap_lambda = 0

        cum_rewards.append(cum_reward)
        bootstrap_obs.append(bootstrap_ob)
        bootstrap_gammas.append(bootstrap_lambda)

    return np.array(cum_rewards, dtype=np.float)[::-1, np.newaxis], \
           np.array(bootstrap_obs)[::-1], \
           np.array(bootstrap_gammas, dtype=np.float)[::-1, np.newaxis]


class TDTrainer(base.Trainer):
    """TDTrainer trainer.

    Trains the network based using temporal difference targets
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        temporal_diff_n,
        gamma=1.0,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
        network_fn=None,
        polyak_coeff=None
    ):
        """Initializes TDTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            temporal_diff_n: temporal difference distance, np.inf is supported
            gamma: discount rate
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
            network_fn: function to create the target network
            polyak_coeff: polyak averaging coefficient
        """
        super().__init__(network_signature)
        target = lambda episode: target_n_return(episode, temporal_diff_n,
                                                 gamma)
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # TODO: possibly better names? (input, target)
        datapoint_sig = (network_signature.input, (network_signature.output,
                                                   network_signature.input,
                                                   network_signature.output))
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy
        self._polyak_coeff = polyak_coeff
        if self._polyak_coeff is not None:
            self._target_network = network_fn()
            self._target_network_params = None

    def add_episode(self, episode):
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(
            (
                episode.transition_batch.observation,
                self._target_fn(episode),
            ),
            buckets,
        )

    def train_epoch(self, network):
        if self._polyak_coeff is not None:
            current_params = network.params
            if self._target_network_params is None:
                self._target_network_params = current_params

            # TODO(pm): this works when nn parameters are given by
            # a list np.arrays, e.g keras nn. Make it work for otherwise
            target_network_params_ = []
            for target_nn_layer, current_nn_layer in \
                    zip(self._target_network_params, current_params):
                target_network_params_.append(
                    self._polyak_coeff * target_nn_layer +
                    (1.0 - self._polyak_coeff) * current_nn_layer)

            self._target_network_params = target_network_params_
            self._target_network.params = self._target_network_params
        else:
            self._target_network = network

        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                obs, (cum_reward, bootstrap_obs, bootstrap_gamma) = \
                    self._replay_buffer.sample(self._batch_size)
                preds = self._target_network.predict(bootstrap_obs)
                target = cum_reward + bootstrap_gamma * preds
                yield obs, target

        return network.train(data_stream)
