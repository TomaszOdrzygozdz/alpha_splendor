"""Parse and return mrunner gin-config and set-up Neptune."""

import atexit
import datetime
import os

import cloudpickle
import neptune

_experiment = None


def get_configuration(spec_path):
    """Get mrunner gin-config overrides and configure the Neptune experiment."""
    global _experiment  # pylint: disable=global-statement

    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise KeyError('To run with mrunner please set your NEPTUNE_API_TOKEN '
                       'environment variable')

    with open(spec_path, 'rb') as f:
        specification = cloudpickle.load(f)
    parameters = specification['parameters']

    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == '@' or value[0] == '%'):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    git_info = specification.get('git_info', None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()

    neptune.init(project_qualified_name=specification['project'])
    # Set pwd property with path to experiment.
    properties = {'pwd': os.getcwd()}
    neptune.create_experiment(name=specification['name'],
                              tags=specification['tags'],
                              params=parameters,
                              properties=properties,
                              git_info=git_info)
    atexit.register(neptune.stop)

    _experiment = neptune.get_experiment()
    return gin_bindings


def log_neptune(name, step, value):
    """Logs a scalar to Neptune."""
    del step
    _experiment.send_metric(name, value)
