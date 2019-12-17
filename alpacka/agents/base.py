"""Agent base classes."""

from alpacka import data


class Agent:
    """Agent base class.

    Agents can use neural networks internally. Network prediction is run outside
    of the Agent, so it can be batched across multiple Agents for efficiency.
    This is done using a coroutine API, explained in solve().
    """

    def __init__(self, action_space):
        """Initializes Agent.

        Args:
            action_space (gym.Space): Action space. It's passed in the
                constructor instead of being inferred from env in solve(),
                because it shouldn't change between environments and this way
                the API for stateless OnlineAgents is simpler.
        """
        self._action_space = action_space

    def solve(self, env, init_state=None):
        """Solves a given environment.

        Coroutine, suspends execution for every neural network prediction
        request. This enables a very convenient interface for requesting
        predictions by the Agent:

            def solve(self, env, init_state=None):
                # Planning...
                predictions = yield inputs
                # Planning...
                predictions = yield inputs
                # Planning...
                return episode

        Example usage:

            coroutine = agent.solve(env)
            try:
                prediction_request = next(coroutine)
                network_output = process_request(prediction_request)
                prediction_request = coroutine.send(network_output)
                # Possibly more prediction requests...
            except StopIteration as e:
                episode = e.value

        Agents that do not use neural networks should wrap their solve() method
        in an @asyncio.coroutine decorator, so Python knows to treat it as
        a coroutine even though it doesn't have any yield.

        Args:
            env (gym.Env): Environment to solve.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            (Agent/Trainer-specific) Episode object summarizing the collected
            data for training the Network.
        """
        raise NotImplementedError


class OnlineAgent(Agent):
    """Base class for online agents, i.e. planning on a per-action basis.

    Provides a default implementation of Agent.solve(), returning a Transition
    object with the collected batch of transitions.
    """

    def reset(self, env):
        """Resets the agent state.

        Called for every new environment to be solved. Overriding is optional.

        Args:
            env (gym.Env): Environment to solve.
        """

    def act(self, observation):
        """Determines the next action to be performed.

        Coroutine, suspends execution similarly to Agent.solve().

        In model-based agents, the original environment state MUST be restored
        in the end of act(). This is not checked at runtime, since it would be
        a big overhead for heavier environments.

        Args:
            observation (Env-dependent): Observation from the environment.

        Yields:
            A stream of Network inputs requested for inference.

        Returns:
            Action to make in the environment.
        """
        raise NotImplementedError

    def solve(self, env, init_state=None):
        """Solves a given environment using OnlineAgent.act().

        Args:
            env (gym.Env): Environment to solve.
            init_state (object): Reset the environment to this state.
                If None, then do normal gym.Env.reset().

        Yields:
            Network-dependent: A stream of Network inputs requested for
            inference.

        Returns:
            data.Episode: Episode object containing a batch of collected
            transitions and the return for the episode.
        """
        self.reset(env)

        if init_state is None:
            # Model-free case...
            observation = env.reset()
        else:
            # Model-based case...
            observation = env.restore_state(init_state)

        transitions = []
        done = False
        info = {}
        while not done:
            # Forward network prediction requests to BatchStepper.
            action = yield from self.act(observation)
            (next_observation, reward, done, info) = env.step(action)

            transitions.append(data.Transition(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
            ))
            observation = next_observation

        return_ = sum(transition.reward for transition in transitions)
        solved = info['solved'] if 'solved' in info else None
        transition_batch = data.nested_stack(transitions)
        return data.Episode(
            transition_batch=transition_batch,
            return_=return_,
            solved=solved,
        )