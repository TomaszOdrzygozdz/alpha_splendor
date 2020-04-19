"""Dummy Trainer for own_testing."""

import gin

from alpacka.trainers import base


@gin.configurable
class DummyTrainer(base.Trainer):
    """Dummy Trainer for own_testing and use with plain Networks (not trainable)."""

    def add_episode(self, episode):
        del episode

    def train_epoch(self, network):
        return {}
