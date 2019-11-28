"""Network interface and its implementations."""

import gin

from planning.networks import core
from planning.networks import keras


# Configure networks in this module to ensure they're accessible via the
# planning.networks.* namespace.
def configure_network(network_class):
    return gin.external_configurable(
        network_class, module='planning.networks'
    )


Network = core.Network  # pylint: disable=invalid-name
DummyNetwork = configure_network(core.DummyNetwork)  # pylint: disable=invalid-name
KerasNetwork = configure_network(keras.KerasNetwork)  # pylint: disable=invalid-name
