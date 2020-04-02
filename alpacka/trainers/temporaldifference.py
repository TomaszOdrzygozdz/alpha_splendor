"""Supervised trainer."""

import gin
import numpy as np

from alpacka import data
from alpacka.trainers import base
from alpacka.trainers import replay_buffers


@gin.configurable
def target_n_return(episode):
    x = np.cumsum(episode.transition_batch.reward[::-1], dtype=np.float)[::-1, np.newaxis]
    return x


class TDTrainer(base.Trainer):
    """TDTrainer trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        network_signature,
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
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(network_signature)
        target = target_n_return
        self._target_fn = lambda episode: data.nested_map(
            lambda f: f(episode), target
        )
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # (input, target)
        datapoint_sig = (network_signature.input, network_signature.output)
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
                yield self._replay_buffer.sample(self._batch_size)

        return network.train(data_stream)
