"""Entrypoint of the experiment."""

import itertools

from gym.envs import classic_control

from planning import agents
from planning import batch_steppers
from planning import networks
from planning import trainers


class Runner:
    """Main class running the experiment."""

    def __init__(
        self,
        env_class=classic_control.CartPoleEnv,
        agent_class=agents.RandomAgent,
        network_class=networks.DummyNetwork,
        n_envs=16,
        collect_real=True,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        trainer_class=trainers.DummyTrainer,
        n_epochs=None,
    ):
        """Initializes the runner.

        Args:
            n_epochs: (int or None) Number of epochs to run for, or indefinitely
                if None.
        """
        self._batch_stepper = batch_stepper_class(
            env_class=env_class,
            agent_class=agent_class,
            network_class=network_class,
            n_envs=n_envs,
            collect_real=collect_real,
        )
        self._network = network_class()
        self._trainer = trainer_class(self._network)
        self._n_epochs = n_epochs
        self._epoch = 0

    def run_epoch(self):
        """Runs a single epoch."""
        transition_batch = self._batch_stepper.run_episode_batch(
            self._network.params
        )
        self._trainer.add_transition_batch(transition_batch)
        self._trainer.train_epoch()
        self._epoch += 1

    def run(self):
        """Runs the main loop."""
        if self._n_epochs is None:
            epochs = itertools.repeat(None)  # Infinite stream of Nones.
        else:
            epochs = range(self._n_epochs)

        for _ in epochs:
            self.run_epoch()


if __name__ == '__main__':
    runner = Runner()
    runner.run()
