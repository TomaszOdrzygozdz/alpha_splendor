"""Replay buffer."""

import numpy as np

from planning import data


class ReplayBuffer:
    """Replay buffer.

    Stores datapoints in a queue of fixed size. Adding to a full buffer
    overwrites the oldest ones.
    """

    def __init__(self, capacity):
        """Initializes the replay buffer.

        Args:
            capacity (int): Maximum size of the buffer.
        """
        self._capacity = capacity
        self._size = 0
        self._data_buffer = None
        self._insert_index = 0

    def _init_buffer(self, prototype):
        """Initializes the data buffer.

        Args:
            prototype (pytree): Example datapoint object to infer the structure
                of data from.
        """
        def init_array(array_prototype):
            shape = (self._capacity,) + array_prototype.shape[1:]
            return np.zeros(shape, array_prototype.dtype)
        self._data_buffer = data.nested_map(init_array, prototype)

    def add(self, stacked_datapoints):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
        """
        if self._data_buffer is None:
            self._init_buffer(stacked_datapoints)

        n_elems = data.choose_leaf(data.nested_map(
            lambda x: x.shape[0], stacked_datapoints
        ))

        def insert_to_array(buf, elems):
            buf_size = buf.shape[0]
            assert elems.shape[0] == n_elems
            index = self._insert_index
            # Insert up to buf_size at the current index.
            buf[index:min(index + n_elems, buf_size)] = elems[:buf_size - index]
            # Insert whatever's left at the beginning of the buffer.
            buf[:max(index + n_elems - buf_size, 0)] = elems[buf_size - index:]

        # Insert to all arrays in the pytree.
        data.nested_zip_with(
            insert_to_array, (self._data_buffer, stacked_datapoints)
        )
        self._size = min(self._insert_index + n_elems, self._capacity)
        self._insert_index = (self._insert_index + n_elems) % self._capacity

    def sample(self, batch_size):
        """Samples a batch of datapoints.

        Args:
            batch_size (int): Number of datapoints to sample.

        Returns:
            Datapoint object with sampled datapoints stacked along the 0 axis.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._data_buffer is None:
            raise ValueError('Cannot sample from an empty buffer.')
        indices = np.random.randint(self._size, size=batch_size)
        return data.nested_map(lambda x: x[indices], self._data_buffer)
