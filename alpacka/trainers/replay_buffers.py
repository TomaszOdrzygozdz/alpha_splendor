"""Uniform replay buffer."""

import numpy as np

from alpacka import data


class UniformReplayBuffer:
    """Replay buffer with uniform sampling.

    Stores datapoints in a queue of fixed size. Adding to a full buffer
    overwrites the oldest ones.
    """

    def __init__(self, datapoint_spec, capacity):
        """Initializes the replay buffer.

        Args:
            datapoint_spec (pytree): Pytree of shape tuples, defining the
                structure of data to be stored.
            capacity (int): Maximum size of the buffer.
        """
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0

        def init_array(shape):
            shape = (self._capacity,) + shape
            return np.zeros(shape)
        self._data_buffer = data.nested_map(
            init_array, datapoint_spec,
            # datapoint_spec has shape tuples at leaves, we don't want to map
            # over them so we stop one level higher.
            stop_fn=data.is_last_level,
        )

    def add(self, stacked_datapoints):
        """Adds datapoints to the buffer.

        Args:
            stacked_datapoints (pytree): Transition object containing the
                datapoints, stacked along axis 0.
        """
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
