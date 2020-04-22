import sys
from typing import List

import atexit
import functools
import gin
import os
import neptune

from alpacka import metric_logging
from alpacka.batch_steppers import ray

from io import StringIO


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


@gin.configurable
class Experiment:
    def __init__(self,
                 name: str = None,
                 project_qualified_name: str = None,
                 description = '',
                 tags: List[str] = None,
                 extra_params = None):

        assert name is not None, 'You must provide experiment name'
        assert project_qualified_name is not None, 'You must provide project qualified name'
        self.name = name
        self.project_qualified_name = project_qualified_name
        self.description = description
        self.tags = tags if tags is not None else []
        machine_tag = 'cluster-run' if self._detect_cluster() else 'local-run'
        self.tags.append(machine_tag)
        self.params = {}

    def parse_params_from_gin_config(self, config_file):
        def row_to_param(row):
            if '#' in row:
                return None, None
            key_value_list = row.split('=')
            if len(key_value_list) != 2:
                return None, None
            key = key_value_list[0].strip("\n '")
            value = key_value_list[1].strip("\n '")
            return key, value

        with open(config_file[0], 'r') as file:
            for row in file:
                key, value = row_to_param(row)
                if key is not None:
                    self.params[key] = value

    def _detect_cluster(self):
        host_name = os.popen('hostname').read()
        return not host_name == 'tomasz-LAPTOP\n'


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, experiment):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self._experiment = experiment

    def log_scalar(self, name, step, value):
        """Logs a scalar to Neptune."""
        self._experiment.send_metric(name, step, value)

    def log_image(self, name, step, img):
        """Logs an image to Neptune."""
        self._experiment.send_image(name, step, img)

    def log_property(self, name, value):
        """Logs a property to Neptune."""
        self._experiment.set_property(name, value)


class NeptuneAPITokenException(Exception):
    def __init__(self):
        super().__init__('NEPTUNE_API_TOKEN environment variable is not set!')


def configure_neptune(experiment: Experiment, cluster_config = None):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise NeptuneAPITokenException()


    neptune.init(project_qualified_name=experiment.project_qualified_name)
    # Set pwd property with path to experiment.
    properties = {'pwd': os.getcwd()}
    with Capturing() as neptune_link:
        neptune.create_experiment(name=experiment.name,
                                  tags=experiment.tags,
                                  params=experiment.params,
                                  description= experiment.description,
                                  properties=properties,
                                  upload_stdout=False)
    atexit.register(neptune.stop)

    # Add hook for Ray workers to make  them connect with appropriate neptune
    # experiment and set neptune logger.
    def connect_to_neptune_experiment_add_logger(project_id, experiment_id):
        neptune.init(project_id)
        exp = neptune.project.get_experiments(
            id=experiment_id
        )[0]
        metric_logging.register_logger(NeptuneLogger(exp))

    ray.register_worker_init_hook(
        functools.partial(
            connect_to_neptune_experiment_add_logger,
            project_id=neptune.project.full_id,
            experiment_id=neptune.get_experiment().id,
        )
    )

    return NeptuneLogger(neptune.get_experiment()), neptune_link[0]
