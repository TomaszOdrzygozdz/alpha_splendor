""" Utilities for Neptune."""

import neptune

from alpacka import metric_logging


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, experiment):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self._experiment = experiment

    def log_scalar(self, name, step, value):
        """Logs a scalar to Neptune."""
        del step
        self._experiment.send_metric(name, value)

    def log_image(self, name, step, img):
        """Logs an image to Neptune."""
        del step
        self._experiment.send_image(name, img)

    def log_property(self, name, value):
        """Logs a property to Neptune."""
        self._experiment.set_property(name, value)


def connect_to_neptune_experiment_add_logger(neptune_info):
    """Connect to existing neptune experiment and set logger.

    Args:
        neptune_info: dictionary with "project_full_id", "experiment_id"
            allowing to connect to existing neptune experiment, or None
    """
    if neptune_info:
        neptune.init(neptune_info['project_full_id'])
        exp = neptune.project.get_experiments(
            id=neptune_info['experiment_id']
        )[0]
        metric_logging.register_logger(NeptuneLogger(exp))
