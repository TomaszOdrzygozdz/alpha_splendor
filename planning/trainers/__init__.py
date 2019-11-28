"""Neural network trainers."""

import gin

from planning.trainers import dummy
from planning.trainers import supervised
from planning.trainers.base import Trainer


# Configure trainers in this module to ensure they're accessible via the
# planning.trainers.* namespace.
def configure_trainer(trainer_class):
    return gin.external_configurable(
        trainer_class, module='planning.trainers'
    )


DummyTrainer = configure_trainer(dummy.DummyTrainer)  # pylint: disable=invalid-name
SupervisedTrainer = configure_trainer(supervised.SupervisedTrainer)  # pylint: disable=invalid-name
