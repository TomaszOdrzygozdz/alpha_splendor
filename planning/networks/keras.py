"""Network interface implementation using the Keras framework."""

import tensorflow as tf
from tensorflow import keras

from planning import networks


class KerasNetwork(networks.NetworkFactory, networks.Network):
    """Network implementation in Keras.

    Args:
        model_fn: It should take an input shape and return tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(self, model_fn, optimizer, loss, metrics=None,
                 train_callbacks=None, **kwargs):

        self.model_fn = model_fn
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.train_callbacks = train_callbacks or []
        self.compile_kwargs = kwargs

        self._model = None
        self._data_types = None

    def construct_network(self, input_shape):
        """Constructs Network implementation.

        Args:
            input_shape: tf.keras.Model input shape.

        Return:
            Network implementation.
        """
        self._model = self.model_fn(input_shape)
        self._model.compile(optimizer=self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics,
                            **self.compile_kwargs)

        self._data_types = (
            self._model.input.dtype,
            self._model.output.dtype,
        )

        return self

    def train(self, data_stream):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
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
        return self._model.fit_generator(dataset, epochs=1, verbose=0,
                                         callbacks=self.train_callbacks)

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.
        """

        return self._model.predict_on_batch(inputs)

    @property
    def params(self):
        """Returns network parameters."""

        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self._model.load_weights(checkpoint_path)


def mlp(input_shape, hidden_sizes=(32,), activation='relu',
        output_activation=None, **kwargs):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for h in hidden_sizes[:-1]:
        x = keras.layers.Dense(h, activation=activation)(x)
    outputs = keras.layers.Dense(hidden_sizes[-1], activation=output_activation,
                                 name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)
