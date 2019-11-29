"""Network interface implementation using the Keras framework."""

import gin

import tensorflow as tf
from tensorflow import keras

from planning.networks import core


@gin.configurable
def mlp(input_shape, hidden_sizes=(32,), activation='relu',
        output_activation=None):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)
    outputs = keras.layers.Dense(
        # 1 output hardcoded for now (value networks).
        # TODO(koz4k): Lift this restriction.
        1,
        activation=output_activation,
        name='predictions',
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


class KerasNetwork(core.Network):
    """Network implementation in Keras.

    Args:
        model_fn: It should take an input shape and return tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **compile_kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(
        self,
        input_shape,
        model_fn=mlp,
        optimizer='adam',
        loss='mean_squared_error',
        metrics=None,
        train_callbacks=None,
        **compile_kwargs
    ):
        super().__init__(input_shape)
        self._model = model_fn(input_shape)
        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics or [],
                            **compile_kwargs)

        self.train_callbacks = train_callbacks or []

    def train(self, data_stream):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
                the updates on.

        Returns:
            dict: Collected metrics, indexed by name.
        """

        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=(self._model.input.dtype, self._model.output.dtype)
        )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        history = self._model.fit_generator(dataset, epochs=1, verbose=0,
                                            callbacks=self.train_callbacks)
        # history contains epoch-indexed sequences. We run only one epoch, so
        # we take the only element.
        return {name: values[0] for (name, values) in history.history.items()}

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.

        Returns:
            Agent-dependent: Network predictions.
        """

        return self._model.predict_on_batch(inputs).numpy()

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
