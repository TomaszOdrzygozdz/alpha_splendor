"""Batch steppers."""

import gin

from alpacka.batch_steppers import local
from alpacka.batch_steppers import ray
import os

# Configure agents in this module to ensure they're accessible via the
# alpacka.batch_steppers.* namespace.
def configure_batch_stapper(batch_stepper_class, name=None):
    return gin.external_configurable(
        batch_stepper_class, module='alpacka.batch_steppers', name=name
    )

if "LOCAL_RUN" in os.environ:
    LocalBatchStepper = configure_batch_stapper(local.LocalBatchStepper,
                                                "AutoBatchStepper")  # pylint: disable=invalid-name
    RayBatchStepper = configure_batch_stapper(ray.RayBatchStepper)  # pylint: disable=invalid-name
else:
    LocalBatchStepper = configure_batch_stapper(local.LocalBatchStepper)  # pylint: disable=invalid-name
    RayBatchStepper = configure_batch_stapper(ray.RayBatchStepper,
                                              "AutoBatchStepper")  # pylint: disable=invalid-name