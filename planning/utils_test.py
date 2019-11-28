import numpy as np
from planning import utils


def test_dataset_batcher():
    # Set up
    x_batches = [
        [[1, -1], [2, -2]],
        [[3, -3], [4, -4]],
        [[5, -5]]]
    y_batches = [
        [10, 20],
        [30, 40],
        [50]]
    x_data = np.array([[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]])
    y_data = np.array([10, 20, 30, 40, 50])

    batcher = utils.DatasetBatcher(x_data, y_data, 2, False)

    # Run and Test
    for x_gt, y_gt, (x, y) in zip(x_batches, y_batches, batcher):
        assert np.all(x_gt == x)
        assert np.all(y_gt == y)
