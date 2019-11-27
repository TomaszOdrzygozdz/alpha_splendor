"""Entrypoint of the experiment."""

import argparse
import itertools
import os

import gin

from planning import agents
from planning import batch_steppers
from planning import envs
from planning import metric_logging
from planning import networks
from planning import trainers


@gin.configurable
class Runner:
    """Main class running the experiment."""

    def __init__(
        self,
        output_dir,
        env_class=envs.CartPole,
        agent_class=agents.RandomAgent,
        network_class=networks.DummyNetwork,
        n_envs=16,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        trainer_class=trainers.DummyTrainer,
        n_epochs=None,
    ):
        """Initializes the runner.

        Args:
            n_epochs: (int or None) Number of epochs to run for, or indefinitely
                if None.
        """
        self._output_dir = os.path.expanduser(output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

        self._batch_stepper = batch_stepper_class(
            env_class=env_class,
            agent_class=agent_class,
            network_class=network_class,
            n_envs=n_envs,
        )
        self._network = network_class()
        self._trainer = trainer_class(self._network)
        self._n_epochs = n_epochs
        self._epoch = 0

    def _log_metrics(self, episodes):
        return_mean = sum(
            episode.return_ for episode in episodes
        ) / len(episodes)
        metric_logging.log_scalar('return_mean', self._epoch, return_mean)

    def _save_gin(self):
        # TODO(koz4k): Send to neptune as well.
        config_path = os.path.join(self._output_dir, 'config.gin')
        with open(config_path, 'w') as f:
            f.write(gin.operative_config_str())

    def run_epoch(self):
        """Runs a single epoch."""
        episodes = self._batch_stepper.run_episode_batch(
            self._network.params
        )
        self._log_metrics(episodes)
        for episode in episodes:
            self._trainer.add_episode(episode)
        self._trainer.train_epoch()

        if self._epoch == 0:
            # Save gin operative config into a file. "Operative" means the part
            # that is actually used in the experiment. We need to run an epoch
            # first, so gin can figure that out.
            self._save_gin()

        self._epoch += 1

    def run(self):
        """Runs the main loop."""
        if self._n_epochs is None:
            epochs = itertools.repeat(None)  # Infinite stream of Nones.
        else:
            epochs = range(self._n_epochs)

        for _ in epochs:
            self.run_epoch()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    parser.add_argument(
        '--config_file', action='append', help='Gin config files.'
    )
    parser.add_argument(
        '--config', action='append', help='Gin config overrides.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    gin.parse_config_files_and_bindings(args.config_file, args.config)

    runner = Runner(args.output_dir)
    runner.run()
