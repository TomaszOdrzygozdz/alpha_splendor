"""Alpacka initialization."""

import gin

from alpacka import utils

gin.config._OPERATIVE_CONFIG_LOCK = utils.SerializableLock()  # pylint: disable=protected-access
