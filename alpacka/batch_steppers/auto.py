"""Environment steppers."""

import os
from alpacka.batch_steppers import LocalBatchStepper, RayBatchStepper

if "LOCAL_RUN" in os.environ:
    class AutoBatchStepper(LocalBatchStepper):
        pass
else:
    class AutoBatchStepper(RayBatchStepper):
        pass
