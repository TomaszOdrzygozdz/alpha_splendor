"""Utils."""

import gin

from alpacka.utils import dask

# Configure utils in this module to ensure they're accessible via the
# alpacka.utils.* namespace.


def configure_util(util_class):
    return gin.external_configurable(
        util_class, module='alpacka.utils'
    )


SerializableLock = configure_util(dask.SerializableLock)  # pylint: disable=invalid-name
