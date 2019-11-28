import random

import numpy as np


class DatasetBatcher:
    """Creates batches of data and target from a dataset.

    It implements tf.keras.utils.Sequence interface and is iterable."""

    def __init__(self, x_set, y_set, batch_size=16, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._index = []

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def __iter__(self):
        self._index = list(range(len(self)))
        if self.shuffle:
            random.shuffle(self._index)
        else:
            self._index.reverse()
        return self

    def __next__(self):
        if len(self._index) > 0:
            return self[self._index.pop()]
        else:
            raise StopIteration
