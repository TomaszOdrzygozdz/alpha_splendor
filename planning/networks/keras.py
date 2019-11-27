"""Network interface implementation using the Keras framework."""

import tensorflow as tf
from tensorflow import keras

from planning.networks import Network


class KerasNetwork(Network):
    """Network implementation in Keras.

    Args:
        model: Not compiled tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        metrics: See tf.keras.Model.compile docstring for possible values.
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        ds_proc_fn: Allows for modifications to training dataset before
            training. Should take tf.data.Dataset and return tf.data.Dataset
            (Default: None).
        **kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(self, model, optimizer, loss, metrics=None,
                 train_callbacks=None, ds_proc_fn=None, **kwargs):

        self.model = model
        self.train_callbacks = train_callbacks or []
        self.ds_proc_fn = ds_proc_fn

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics or [],
                           **kwargs)

        self._data_types = (
            self.model.input.dtype,
            self.model.output.dtype,
        )
        self._data_shapes = (
            self.model.input.shape[1:],
            self.model.output.shape[1:],
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

        dataset = self._get_train_dataset(data_stream)
        return self.model.fit(dataset, epochs=1, verbose=0,
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

    def _get_train_dataset(self, data_stream):
        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=self._data_types,
            output_shapes=self._data_shapes
        )

        if self.ds_proc_fn is not None:
            dataset = self.ds_proc_fn(dataset)

        return dataset


def define_keras_mlp(input_shape, hidden_sizes=(32,), activation='relu',
                     output_activation=None):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for h in hidden_sizes[:-1]:
        x = keras.layers.Dense(h, activation=activation)(x)
    outputs = keras.layers.Dense(hidden_sizes[-1], activation=output_activation,
                                 name='predictions')(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def get_ds_batch_n_shuffle_fn(batch_size, buffer_size, seed=None):
    def _ds_batch_n_shuffle_fn(dataset):
        return dataset.shuffle(buffer_size, seed).batch(batch_size)
    return _ds_batch_n_shuffle_fn
