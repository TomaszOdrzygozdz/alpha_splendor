"""Monte Carlo Tree Search agent."""

import gym

from planning.agents import base
from planning.agents import core


def rate_new_leaves_with_rollouts(
    leaf,
    observation,
    model,
    discount,
    rollout_agent_class=core.RandomAgent,
    rollout_time_limit=100,
):
    agent = rollout_agent_class(model.action_space)
    init_state = model.clone_state()

    child_qualities = []
    for init_action in range(model.action_space.n):
        (observation, reward, done, _) = model.step(init_action)
        quality = reward
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            action = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            total_discount *= discount
            quality += total_discount * reward
            time += 1
        child_qualities.append(quality)
        model.restore_state(init_state)
    return child_qualities


class TreeNode:
    """Node of the search tree.

    Attrs:
        quality: Instead of value, so we can handle rating with both V-networks
            and Q-networks. Quality(s, a) = reward(s, a) + discount * value(s').
    """

    def __init__(self, init_quality):
        self._init_quality = init_quality
        self._reward_sum = 0
        self._reward_count = 0
        self.graph_node = None
        self.children = None

    def visit(self, reward, value):
        assert self.graph_node is not None, 'Graph node must be assigned first.'
        self._reward_sum += reward
        self._reward_count += 1
        self.graph_node.visit(value)

    def quality(self, discount):
        if self._reward_count == 0:
            return self._init_quality
        return (
            self._init_quality + self._reward_sum +
            discount * self.graph_node.value * self._reward_count
        ) / (self._reward_count + 1)

    @property
    def is_leaf(self):
        return self.children is None


class GraphNode:
    """Node of the search graph.

    In the graph mode, corresponds to a state in the MDP. Outside of the graph
    mode, corresponds 1-1 to a TreeNode.
    """

    def __init__(self):
        self._value_sum = 0
        self._value_count = 0

    def visit(self, value):
        self._value_sum += value
        self._value_count += 1

    @property
    def value(self):
        return self._value_sum / self._value_count


class MCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(
        self,
        n_passes=10,
        discount=0.99,
        rate_new_leaves_fn=rate_new_leaves_with_rollouts,
        graph_mode=False,
    ):
        """Initializes MCTSAgent.

        Args:
            n_passes: (int) Number of MCTS passes per act().
            discount: (float) Discount factor.
            rate_new_leaves: Coroutine estimating qualities of new leaves. Can
                ask for predictions using a Network. Should return qualities for
                every child of a given leaf node.
                Signature: (leaf, observation, model, discount) -> [float].
            graph_mode: (bool) Turns on using transposition tables, turning the
                search graph from a tree to a DAG.
        """
        self._n_passes = n_passes
        self._discount = discount
        self._rate_new_leaves = rate_new_leaves_fn
        self._graph_mode = graph_mode
        self._model = None
        self._root = None
        self._root_state = None
        self._state_to_graph_node = {}

    def _rate_children(self, node):
        return [
            node.children[action].quality(self._discount)
            for action in range(len(node.children))
        ]

    def _choose_action(self, node):
        # TODO(koz4k): Distinguish exploratory/not.
        child_qualities = self._rate_children(node)
        (_, action) = max(zip(child_qualities, range(len(child_qualities))))
        return action

    def _traverse(self, root, observation):
        """Chooses a path from the root to a leaf in the search tree.

        Returns:
            Pair (observation, done, path) where path is a list of pairs
            (reward, node), including leaf, such that reward is obtained when
            transitioning _to_ node. In case of a "done" state, traversal is
            interrupted.
        """
        path = []
        path.append((0, root))
        node = root
        done = False
        while not node.is_leaf and not done:
            action = self._choose_action(node)
            node = node.children[action]
            (observation, reward, done, _) = self._model.step(action)
            path.append((reward, node))
        return (observation, done, path)

    def _expand_leaf(self, leaf, observation, done):
        """Expands a leaf and returns its quality.

        The leaf's new children are assigned initial qualities, but they're not
        backpropagated yet. They will be only when we expand those new leaves.

        Returns:
            Quality of the expanded leaf, or None if the pass should be
            interrupted because the leaf is a previously-visited node.
        """
        assert leaf.is_leaf
        # TODO(koz4k): Check for loops here.
        # TODO(koz4k): Handle the case when the expanded leaf is on the path.
        if self._graph_mode:
            state = self._model.clone_state()
            graph_node = self._state_to_graph_node.get(state, None)
            if graph_node is None:
                graph_node = GraphNode()
                self._state_to_graph_node[state] = graph_node
            else:
                # Leaf is a previously visited node, interrupt the pass.
                return None
        else:
            graph_node = GraphNode()
        leaf.graph_node = graph_node

        if not done:
            child_qualities = yield from self._rate_new_leaves(
                leaf, observation, self._model, self._discount
            )
            leaf.children = [TreeNode(quality) for quality in child_qualities]
            action = self._choose_action(leaf)
            return child_qualities[action]
        else:
            # In a "done" state, cumulative future return is 0.
            return 0

    def _backpropagate(self, value, path):
        for (reward, node) in reversed(path):
            node.visit(reward, value)
            value = reward + self._discount * value

    def _run_pass(self, root, observation):
        (observation, done, path) = self._traverse(root, observation)
        (_, leaf) = path[-1]
        try:
            quality = yield from self._expand_leaf(leaf, observation, done)
            if quality is None:
                # Leaf is a previously visited node, interrupt the pass.
                return
            self._backpropagate(quality, path)
        finally:
            # Go back to the root state.
            self._model.restore_state(self._root_state)

    def reset(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        self._model = env
        # Initialize root with some quality to avoid division by zero.
        self._root = TreeNode(init_quality=0)

    def act(self, observation):
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self._n_passes):
            yield from self._run_pass(self._root, observation)
        action = self._choose_action(self._root)
        self._root = self._root.children[action]
        return action
