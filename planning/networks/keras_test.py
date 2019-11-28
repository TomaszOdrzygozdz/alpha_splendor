import os

import numpy as np
import pytest
import tempfile
import tensorflow as tf
from tensorflow import keras

from planning import utils
from planning.networks.keras import define_keras_mlp, KerasNetwork  # noqa: E501


@pytest.fixture
def keras_mlp():
    model = define_keras_mlp(input_shape=(784,),
                             hidden_sizes=(16, 10),
                             output_activation='softmax')
    network = KerasNetwork(model,
                           optimizer=keras.optimizers.RMSprop(),
                           loss=keras.losses.CategoricalCrossentropy(),
                           metrics=[keras.metrics.CategoricalAccuracy()])
    return network


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


def test_keras_mlp_save_n_restore_weights(keras_mlp):
    # Set up, Run and Test
    with tempfile.NamedTemporaryFile() as temp_file:
        assert os.path.getsize(temp_file.name) == 0
        keras_mlp.save(temp_file.name)
        assert os.path.getsize(temp_file.name) > 0
        keras_mlp.restore(temp_file.name)
