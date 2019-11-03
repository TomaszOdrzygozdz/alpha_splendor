"""Deep learning framework-agnostic interface for neural networks."""


class Network:
    """Base class for networks."""

    def train(self, transition_batch):
        """Performs one step of training on a batch of transitions."""
        raise NotImplementedError

    def predict(self, inputs):
        """Returns the prediction for a given input."""
        raise NotImplementedError

    @property
    def params(self):
        """Returns network parameters."""
        raise NotImplementedError

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""
        raise NotImplementedError

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""
        raise NotImplementedError


class DummyNetwork(Network):
    """Dummy Network for testing."""

    def __init__(self, predict_output=None):
        self.predict_output = predict_output

    def train(self, transition_batch):
        del transition_batch

    def predict(self, inputs):
        del inputs
        return self.predict_output

    @property
    def params(self):
        return None

    @params.setter
    def params(self, new_params):
        del new_params

    def save(self, checkpoint_path):
        del checkpoint_path

    def restore(self, checkpoint_path):
        del checkpoint_path
