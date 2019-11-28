"""Network interface implementation using the Keras framework."""

import tensorflow as tf
from tensorflow import keras

from planning import networks


class KerasNetwork(networks.Network):
    """Network implementation in Keras.

    Args:
        model: Not compiled tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        metrics: See tf.keras.Model.compile docstring for possible values.
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(self, model, optimizer, loss, metrics=None,
                 train_callbacks=None, **kwargs):

        self.model = model
        self.train_callbacks = train_callbacks or []

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics or [],
                           **kwargs)

        self._data_types = (
            self.model.input.dtype,
            self.model.output.dtype,
        )

    def train(self, data_stream):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of examples to run
                the updates on.

        Returns:
            A History object. Its History.history attribute is a record of
            training loss values and metrics values at successive epochs.
        """

        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=self._data_types
        )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        return self.model.fit_generator(dataset, epochs=1, verbose=0,
                                        callbacks=self.train_callbacks)

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.
        """

        return self.model.predict_on_batch(inputs)

    @property
    def params(self):
        """Returns network parameters."""

        return self.model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self.model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self.model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self.model.load_weights(checkpoint_path)


def mlp(input_shape, hidden_sizes=(32,), activation='relu',
        output_activation=None):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for h in hidden_sizes[:-1]:
        x = keras.layers.Dense(h, activation=activation)(x)
    outputs = keras.layers.Dense(hidden_sizes[-1], activation=output_activation,
                                 name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)
