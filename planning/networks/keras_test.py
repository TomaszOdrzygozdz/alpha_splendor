import os

import numpy as np
import pytest
import tempfile
import tensorflow as tf
from tensorflow import keras

from planning import utils
from planning.networks import keras as keras_network


@pytest.fixture
def keras_mlp_factory():
    def _model_fn(input_shape):
        return keras_network.mlp(input_shape=input_shape,
                                 hidden_sizes=(16, 10),
                                 output_activation='softmax')

    network_factory = keras_network.KerasNetwork(
        model_fn=_model_fn,
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()])

    return network_factory


@pytest.fixture
def keras_mlp(keras_mlp_factory):
    # NOTE: This is how you should create networks! Take its factory as param
    #       and construct it with input shape.
    mnist_input_shape = (784,)
    return keras_mlp_factory.construct_network(mnist_input_shape)


def test_keras_mlp_train_epoch_on_mnist(keras_mlp):
    # Set up
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train[:100]
    y_train = y_train[:100]

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    y_train = tf.one_hot(y_train, depth=10, dtype=tf.float32)

    def mnist_data_stream():
        return utils.DatasetBatcher(x_train, y_train, batch_size=16)

    # Run
    history = keras_mlp.train(mnist_data_stream)

    # Test
    assert 'loss' in history.history
    assert 'categorical_accuracy' in history.history


def test_keras_mlp_predict_batch_on_mnist(keras_mlp):
    # Set up
    (data, _), _ = keras.datasets.mnist.load_data()
    data_batch = data[:16].reshape(-1, 784).astype('float32') / 255

    # Run
    pred_batch = keras_mlp.predict(data_batch)

    # Test
    assert pred_batch.shape == (16, 10)


def test_keras_mlp_modify_weights(keras_mlp):
    # Set up
    new_params = keras_mlp.params
    for p in new_params:
        p *= 2

    # Run
    keras_mlp.params = new_params

    # Test
    for new, mlp in zip(new_params, keras_mlp.params):
        assert np.all(new == mlp)


def test_keras_mlp_save_weights(keras_mlp):
    # Set up, Run and Test
    with tempfile.NamedTemporaryFile() as temp_file:
        assert os.path.getsize(temp_file.name) == 0
        keras_mlp.save(temp_file.name)
        assert os.path.getsize(temp_file.name) > 0


def test_keras_mlp_restore_weights(keras_mlp):
    with tempfile.NamedTemporaryFile() as temp_file:
        # Set up
        orig_params = keras_mlp.params
        keras_mlp.save(temp_file.name)

        new_params = keras_mlp.params
        for p in new_params:
            p *= 2
        keras_mlp.params = new_params

        # Run
        keras_mlp.restore(temp_file.name)

        # Test
        for orig, mlp in zip(orig_params, keras_mlp.params):
            assert np.all(orig == mlp)
