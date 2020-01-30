"""Agent wrappers."""

from alpacka import utils


class ParamSchedulerWrapper:
    """Base class for all the parameter scheduling Agent wrappers."""

    def __init__(self, agent, attr_name):
        """Initializes ParamSchedulerWrapper.

        Args:
            agent (Agent): Agent which parameter should be annealed.
            attr_name (str): Attribute name under which parameter is stored.
                Can be recursive e.g. 'distribution.temperature'.
        """
        self._agent = agent
        self._attr_name = attr_name

    def solve(self, env, epoch=None, init_state=None, time_limit=None):
        utils.recursive_setattr(
            self._agent, self._attr_name, self._get_current_value(epoch))

        return_ = yield from self._agent.solve(env,
                                               epoch=epoch,
                                               init_state=init_state,
                                               time_limit=time_limit)
        return return_

    def _get_current_value(self, epoch):
        raise NotImplementedError()

    def __getattr__(self, attr_name):
        return getattr(self._agent, attr_name)


class LinearAnnealingWrapper(ParamSchedulerWrapper):
    """Implement the linear annealing parameter schedule.

    Linear Annealing Wrapper computes a current parameter value and
    transfers it to an agent which chooses the action. The threshold
    value is following a linear function decreasing every epoch (reset).
    """

    def __init__(self, agent, attr_name, max_value, min_value, n_epochs):
        """Initializes LinearAnnealingWrapper.

        Args:
            agent (Agent): Agent which parameter should be annealed.
            attr_name (str): Attribute name under which parameter is stored.
                Can be recursive e.g. 'distribution.temperature'.
            max_value (float): Maximal (starting) parameter value.
            min_value (float): Minimal (final) parameter value.
            n_epochs (int): Across how many epochs parameter should reach from
                its starting to its final value.
        """
        super().__init__(agent, attr_name)

        self._min_value = min_value
        self._slope = - (max_value - min_value) / (n_epochs - 1)
        self._intersect = max_value

    def _get_current_value(self, epoch):
        return max(self._min_value, self._slope * epoch + self._intersect)
