"""Network interface and its implementations."""

import gin

from alpacka.networks import core
from alpacka.networks import keras
from alpacka.networks import tensorflow


# Configure networks in this module to ensure they're accessible via the
# alpacka.networks.* namespace.
def configure_network(network_class):
    return gin.external_configurable(
        network_class, module='alpacka.networks'
    )


DummyNetwork = configure_network(core.DummyNetwork)  # pylint: disable=invalid-name
KerasNetwork = configure_network(keras.KerasNetwork)  # pylint: disable=invalid-name
TFMetaGraphNetwork = configure_network(tensorflow.TFMetaGraphNetwork)  # pylint: disable=invalid-name
