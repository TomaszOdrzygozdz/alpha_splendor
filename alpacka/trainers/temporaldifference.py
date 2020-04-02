"""Supervised trainer."""

import gin
import numpy as np
import math

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers



@gin.configurable
def target_n_return(episode, n, lambda_):
    assert n >= 1, "TD(0) does not make sense"
    truncated = episode.truncated
    rewards = episode.transition_batch.reward
    ep_len = len(episode.transition_batch.observation)
    cum_reward = 0

    # TODO remove me after testes
    for i in range(ep_len-1):
        assert np.sum(np.abs(episode.transition_batch.next_observation[i] - episode.transition_batch.observation[i+1])) == 0, "Just a stupid test"


    cum_rewards = []
    bootstrap_obs = []
    bootstrap_lambdas = []
    for curr_obs_index in reversed(range(ep_len)):
        reward_to_remove = 0 if curr_obs_index + n >= ep_len else math.pow(lambda_, n-1)*rewards[curr_obs_index + n]
        cum_reward -= reward_to_remove
        cum_reward *= lambda_
        cum_reward += rewards[curr_obs_index]

        bootstrap_index = min(curr_obs_index + n, ep_len)
        bootstrap_ob = episode.transition_batch.observation[bootstrap_index - 1]
        bootstrap_lambda = math.pow(lambda_, bootstrap_index - curr_obs_index)
        if bootstrap_index == ep_len and not truncated:
            bootstrap_lambda = 0

        cum_rewards.append(cum_reward)
        bootstrap_obs.append(bootstrap_ob)
        bootstrap_lambdas.append(bootstrap_lambda)


    # x = np.cumsum(episode.transition_batch.reward[::-1], dtype=np.float)[::-1, np.newaxis]
    return np.array(cum_rewards, dtype=np.float)[::-1, np.newaxis], np.array(bootstrap_obs)[::-1], np.array(bootstrap_lambdas, dtype=np.float)[::-1, np.newaxis]
    # return np.array(cum_rewards, dtype=np.float)[::-1, np.newaxis], np.array(bootstrap_lambdas, dtype=np.float)[::-1, np.newaxis]


class TDTrainer(base.Trainer):
    """TDTrainer trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
        temporal_diff_n,
        lambda_=1.0,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
    ):
        """Initializes SupervisedTrainer.

        Args:
            network_signature (pytree): Input signature for the network.
            target (pytree): Pytree of functions episode -> target for
                determining the targets for network training. The structure of
                the tree should reflect the structure of a target.
            temporal_diff_n: temporal difference distance, np.inf is supported
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(network_signature)
        target = lambda episode: target_n_return(episode, temporal_diff_n, lambda_)
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # TODO: possibly better names? (input, target)
        datapoint_sig = (network_signature.input, (network_signature.output, network_signature.input, network_signature.output))
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_sig,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy

    def add_episode(self, episode):
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(
            (
                episode.transition_batch.observation,  # input
                self._target_fn(episode),  # target
            ),
            buckets,
        )

    def train_epoch(self, network):

        def data_stream():
            # calculate new labels in fly
            for _ in range(self._n_steps_per_epoch):
                obs, (cum_reward, bootstrap_obs, bootstrap_lambda) = self._replay_buffer.sample(self._batch_size)
                target = cum_reward + bootstrap_lambda * network.predict(bootstrap_obs)
                yield obs, target

        return network.train(data_stream)
