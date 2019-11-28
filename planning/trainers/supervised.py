"""Supervised trainer."""

import gin
import numpy as np

from planning.trainers import base
from planning.trainers import replay_buffer


@gin.configurable
def target_solved(transition_batch):
    return transition_batch.solved.astype(np.int32)


class SupervisedTrainer(base.Trainer):
    """Supervised trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        target_fn=target_solved,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
    ):
        """Initializes SupervisedTrainer.

        Args:
            target_fn (callable): Function transition_batch -> target for
                determining the target for network training.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
        """
        self._target_fn = target_fn
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch
        self._replay_buffer = replay_buffer.ReplayBuffer(
            capacity=replay_buffer_capacity
        )

    def add_episode(self, episode):
        self._replay_buffer.add((
            episode.transition_batch.observation,  # input
            self._target_fn(episode.transition_batch),  # target
        ))

    def train_epoch(self, network):
        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                yield self._replay_buffer.sample(self._batch_size)

        network.train(data_stream)
