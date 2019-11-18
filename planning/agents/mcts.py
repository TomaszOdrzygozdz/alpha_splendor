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

    child_ratings = []
    for init_action in range(model.action_space.n):
        (observation, init_reward, done, _) = model.step(init_action)
        value = 0
        total_discount = 1
        time = 0
        while not done and time < rollout_time_limit:
            action = yield from agent.act(observation)
            (observation, reward, done, _) = model.step(action)
            value += total_discount * reward
            total_discount *= discount
            time += 1
        child_ratings.append((init_reward, value))
        model.restore_state(init_state)
    return child_ratings


class TreeNode:
    """Node of the search tree."""

    def __init__(self, init_reward, init_value=None):
        self._reward_sum = init_reward
        self._reward_count = 1
        self._init_value = init_value
        self._graph_node = None
        self.children = None
        self.is_terminal = False

    def init_graph_node(self, graph_node=None):
        assert self._graph_node is None, 'Graph node initialized twice.'
        if graph_node is None:
            graph_node = GraphNode(self._init_value)
        self._graph_node = graph_node

    @property
    def graph_node(self):
        return self._graph_node

    def visit(self, reward, value):
        self._reward_sum += reward
        self._reward_count += 1
        if not self.is_terminal and value is not None:
            assert self.graph_node is not None, (
                'Graph node must be assigned first.'
            )
            self.graph_node.visit(value)

    def quality(self, discount):
        """Returns the quality of going into this node in the search tree.

        We use it instead of value, so we can handle dense rewards.
        Quality(s, a) = reward(s, a) + discount * value(s').
        """
        return self._reward_sum / self._reward_count + (
            self._graph_node.value
            if self._graph_node is not None else self._init_value
        )

    @property
    def is_leaf(self):
        return self.children is None


class GraphNode:
    """Node of the search graph.

    In the graph mode, corresponds to a state in the MDP. Outside of the graph
    mode, corresponds 1-1 to a TreeNode.
    """

    def __init__(self, init_value):
        self._value_sum = 0
        self._value_count = 0
        if init_value is not None:
            self.visit(init_value)
        # TODO(koz4k): Move children here?

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
            rate_new_leaves: Coroutine estimating rewards and values of new
                leaves. Can ask for predictions using a Network. Should return
                rewards and values for every child of a given leaf node.
                Signature:
                (leaf, observation, model, discount) -> [(reward, value)].
            graph_mode: (bool) Turns on using transposition tables, turning the
                search graph from a tree to a DAG.
        """
        self.n_passes = n_passes
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
        child_ratings = self._rate_children(node)
        (_, action) = max(zip(child_ratings, range(len(child_ratings))))
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

        The leaf's new children are assigned initial rewards and values. The
        reward and value of the "best" new leaf is then backpropagated.

        Returns:
            Quality of a chosen child of the expanded leaf, or None if we
            shouldn't backpropagate quality beause the node has already been
            visited.
        """
        assert leaf.is_leaf
        
        if done:
            leaf.is_terminal = True
            # In a "done" state, cumulative future return is 0.
            return 0

        # TODO(koz4k): Check for loops here.
        # TODO(koz4k): Handle the case when the expanded leaf is on the path.
        already_visited = False
        if self._graph_mode:
            state = self._model.clone_state()
            graph_node = self._state_to_graph_node.get(state, None)
            leaf.init_graph_node(graph_node)
            if graph_node is not None:
                already_visited = True
            else:
                self._state_to_graph_node[state] = leaf.graph_node
        else:
            leaf.init_graph_node()

        child_ratings = yield from self._rate_new_leaves(
            leaf, observation, self._model, self._discount
        )
        leaf.children = [
            TreeNode(reward, value) for (reward, value) in child_ratings
        ]
        if already_visited:
            # Node has already been visited, don't backpropagate quality.
            return None
        action = self._choose_action(leaf)
        return leaf.children[action].quality(self._discount)

    def _backpropagate(self, value, path):
        for (reward, node) in reversed(path):
            node.visit(reward, value)
            if value is not None:
                value = reward + self._discount * value

    def _run_pass(self, root, observation):
        (observation, done, path) = self._traverse(root, observation)
        (_, leaf) = path[-1]
        quality = yield from self._expand_leaf(leaf, observation, done)
        self._backpropagate(quality, path)
        # Go back to the root state.
        self._model.restore_state(self._root_state)

    def reset(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            'MCTSAgent only works with Discrete action spaces.'
        )
        self._model = env
        # Initialize root with some reward to avoid division by zero.
        self._root = TreeNode(init_reward=0)

    def act(self, observation):
        assert self._model is not None, (
            'MCTSAgent works only in model-based mode.'
        )
        self._root_state = self._model.clone_state()
        for _ in range(self.n_passes):
            yield from self._run_pass(self._root, observation)
        action = self._choose_action(self._root)
        self._root = self._root.children[action]
        return action
