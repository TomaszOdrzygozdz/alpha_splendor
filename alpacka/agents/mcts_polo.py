import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.agents import core


def neighbours(model, init_state):
    old_state = model.clone_state()

    model.restore_state(init_state)

    def step_and_rewind(action):
        (observation, reward, done, info) = model.step(action)
        state = model.clone_state()
        solved = 'solved' in info and info['solved']
        model.restore_state(init_state)
        return (observation, reward, done, solved, state)

    results = zip(*[
        step_and_rewind(action) for action in range(model.action_space.n)
    ])
    model.restore_state(old_state)
    return results


@gin.configurable
class ScalarValueTraits:

    zero = 0.0

    def __init__(self, dead_end_value=-2.0):
        self.dead_end = dead_end_value

    def distill_batch(self, value_batch):
        return np.reshape(value_batch, newshape=-1)


class ValueAccumulator:

    def __init__(self, value, state=None):
        # Creates and initializes with typical add
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def add_auxiliary(self, value):
        """
        Additional value for traversals
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.

        May be non-deterministic.  # TODO(pm): What does it mean?
        """
        raise NotImplementedError

    def index(self, parent_value, action):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):

    def __init__(self, value, state=None, mean_max_coeff=1.0):
        self._sum = 0.0
        self._count = 0
        self._max = value
        self.mean_max_coeff = mean_max_coeff
        self.auxiliary_loss = 0.0
        super().__init__(value, state)

    def add(self, value):
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        return (self._sum / self._count)*self.mean_max_coeff \
               + self._max*(1-self.mean_max_coeff)

    def index(self, parent_value=None, action=None):
        return self.get() + self.auxiliary_loss  # auxiliary_loss alters tree traversal in mcts

    def target(self):
        return self.get()

    def count(self):
        return self._count



class GraphNode(object):
    def __init__(self, value_acc,
                 state=None,
                 terminal=False,
                 solved=False,
                 nedges=4):
        self.value_acc = value_acc
        self.rewards = [None] * nedges
        self.state = state
        self.terminal = terminal
        self.solved = solved

# tree node
class TreeNode(object):
    def __init__(self, node: object) -> object:
        self.node = node
        self.children = {}  # {valid_action: Node}

    @property
    def rewards(self):
        return self.node.rewards

    @property
    def value_acc(self):
        return self.node.value_acc

    @property
    def state(self):
        return self.node.state

    @state.setter
    def state(self, state):
        self.node.state = state

    def expanded(self):
        return True if self.children else False

    @property
    def terminal(self):
        return self.node.terminal

    @property
    def solved(self):
        return self.node.solved

    @terminal.setter
    def terminal(self, terminal):
        self.node.terminal = terminal


@gin.configurable
class MCTSValue(base.OnlineAgent):

    def __init__(self,
                 action_space,
                 node_value_mode='bootstrap',
                 gamma=0.99,
                 value_annealing=1.,
                 num_sampling_moves=0,
                 num_mcts_passes=10,
                 avoid_loops=True,
                 avoid_traversal_loop_coeff=0.0,
                 avoid_history_coeff=0.0,
                 history_process_fn = lambda x, solved: (x, {}),
                 ):
        super().__init__(action_space=action_space)
        self._value_traits = ScalarValueTraits()
        self._gamma = gamma
        self._value_annealing = value_annealing
        self._num_sampling_moves = num_sampling_moves
        self._avoid_loops = avoid_loops
        self._state2node = {}
        self.history = []
        self.avoid_traversal_loop_coeff = avoid_traversal_loop_coeff
        if callable(avoid_history_coeff):
            self.avoid_history_coeff = avoid_history_coeff
        else:
            self.avoid_history_coeff = lambda: avoid_history_coeff
        self._node_value_mode = node_value_mode
        self.history_process_fn = history_process_fn
        assert value_annealing == 1., "Annealing temporarily not supported."  # TODO(pm): reenable
        self._num_mcts_passes = num_mcts_passes

    def run_mcts_pass(self, root: TreeNode) -> None:
        # search_path = list of tuples (node, action)
        # leaf does not belong to search_path (important for not double counting its value)
        leaf, search_path = self.tree_traversal(root)
        value = yield from self.expand_leaf(leaf)
        self._backpropagate(search_path, value)

    def run_one_step(self, root):
        # if root is None (start of new episode) or Terminal (end of episode), initialize new root
        root = yield from self.preprocess(root)

        # perform MCTS passes. each pass = tree traversal + leaf evaluation + backprop
        for _ in range(self._num_mcts_passes):
            yield from self.run_mcts_pass(root)
        next, action = self._select_next_node(root)  # INFO: possible sampling for exploration

        return root, next, action

    def tree_traversal(self, root):
        node = root
        seen_states = set()
        search_path = []
        while node.expanded():
            seen_states.add(node.state)
            node.value_acc.add_auxiliary(self.avoid_traversal_loop_coeff)
            #  Avoiding visited states in the fashion of https://openreview.net/pdf?id=Hyfn2jCcKm

            # INFO: if node Dead End, (new_node, action) = (None, None)
            # INFO: _select_child can SAMPLE an action (to break tie)
            states_to_avoid = seen_states if self._avoid_loops else set()
            new_node, action = self._select_child(node, states_to_avoid)  #
            search_path.append((node, action))
            node = new_node
            if new_node is None:  # new_node is None iff node has no unseen children, i.e. it is Dead End
                break
        # at this point node represents a leaf in the tree (and is None for Dead End).
        # node does not belong to search_path.
        return node, search_path

    def _backpropagate(self, search_path, value):
        # Note that a pair
        # (node, action) can have the following form:
        # (Terminal node, None),
        # (Dead End node, None),
        # (TreeNode, action)
        for node, action in reversed(search_path):
            value = td_backup(node, action, value, self._gamma)  # returns value if action is None
            node.value_acc.add(value)
            node.value_acc.add_auxiliary(-self.avoid_traversal_loop_coeff)

    def _get_value(self, obs, states):
        value = yield np.array(obs)
        return self._value_annealing * value
        # return self._value_annealing * value

    def _initialize_graph_node(self, initial_value, state, done, solved):
        value_acc = ScalarValueAccumulator(initial_value, state)
        new_node = GraphNode(value_acc,
                             state=state,
                             terminal=done,
                             solved=solved,
                             nedges=self._action_space.n)
        self._state2node[state] = new_node  # store newly initialized node in _state2node
        return new_node

    def preprocess(self, root):
        if root is not None and not root.terminal:
            root.value_acc.add_auxiliary(self.avoid_history_coeff())
            return root

        # 'reset' mcts internal variables: _state2node and _model
        # TODO(pm): this should be moved to run_one_episode
        self._state2node = {}
        obs = self._model.reset()
        state = self._model.clone_state()
        value = yield from self._get_value([obs], [state])
        value = value[0]
        new_node = self._initialize_graph_node(initial_value=value, state=state, done=False, solved=False)
        new_root = TreeNode(new_node)
        new_root.value_acc.add_auxiliary(self.avoid_history_coeff())
        return new_root

    def initialize_root(self):
        # TODO(pm): seemingly unused function. Refactor
        raise NotImplementedError("should not happen")
        # 'reset' mcts internal variables: _state2node and _model
        self._state2node = {}
        obs = self._model.reset()
        state = self._model.clone_state()
        value = self._get_value([obs], [state])[0]

        new_node = self._initialize_graph_node(initial_value=value, state=state, done=False, solved=False)
        return TreeNode(new_node)

    def expand_leaf(self, leaf: TreeNode):
        if leaf is None:  # Dead End
            return self._value_traits.dead_end

        if leaf.terminal:  # Terminal state
            return self._value_traits.zero

        # neighbours are ordered in the order of actions: 0, 1, ..., _model.num_actions
        obs, rewards, dones, solved, states = neighbours(self._model, leaf.state)

        value_batch = yield from self._get_value(obs=obs, states=states)

        for idx, action in enumerate(range(self._action_space.n)):
            leaf.rewards[idx] = rewards[idx]
            new_node = self._state2node.get(states[idx], None)
            if new_node is None:
                child_value = value_batch[idx] if not dones[idx] else self._value_traits.zero
                new_node = self._initialize_graph_node(
                    child_value, states[idx], dones[idx], solved=solved[idx]
                )
            leaf.children[action] = TreeNode(new_node)

        return leaf.value_acc.get()

    def _child_index(self, parent, action):
        accumulator = parent.children[action].value_acc
        value = accumulator.index(parent.value_acc, action)
        return td_backup(parent, action, value, self._gamma)

    def _rate_children(self, node, states_to_avoid):
        assert self._avoid_loops or len(states_to_avoid) == 0, "Should not happen. There is a bug."
        return [
            (self._child_index(node, action), action)
            for action, child in node.children.items()
            if child.state not in states_to_avoid
        ]

    # here UNLIKE alphazero, we choose final action from the root according to value
    def _select_next_node(self, root):
        # INFO: below line guarantees that we do not perform one-step loop (may be considered slight hack)
        states_to_avoid = {root.state} if self._avoid_loops else set()
        values_and_actions = self._rate_children(root, states_to_avoid)
        if not values_and_actions:
            # when there are no children (e.g. at the bottom states of ChainEnv)
            return None, None
        # TODO: can we do below more elegantly
        if len(self.history) < self._num_sampling_moves:
            chooser = _softmax_sample
        else:
            chooser = max
        (_, action) = chooser(values_and_actions)
        return root.children[action], action

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid):
        values_and_actions = self._rate_children(node, states_to_avoid)
        if not values_and_actions:
            return None, None
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if value == max_value
        ]
        # INFO: here can be sampling
        if len(argmax) > 1:  # PM: This works faster
            action = np.random.choice(argmax)
        else:
            action = argmax[0]
        return node.children[action], action

    # TODO(pm): refactor me
    def solve(self, env, time_limit):
        self._model = env
        new_root = None
        history = []
        transitions = []
        game_steps = 0

        while True:
            old_root, new_root, action = yield from self.run_one_step(new_root)

            history.append((old_root, action, old_root.rewards[action]))
            transitions.append(data.Transition(
                observation=old_root.state.one_hot, action=action, reward=old_root.rewards[action],
                next_observation=None, done=False, agent_info={},
            ))
            game_steps += 1

            # action required if the end of the game (terminal or step limit reached)
            if new_root.terminal or game_steps >= time_limit:

                game_solved = new_root.solved
                nodes = [elem[0] for elem in history]
                history, evaluator_kwargs = self.history_process_fn(history, game_solved)
                # give each state of the trajectory a value
                values = game_evaluator_new(history, self._node_value_mode, self._gamma, game_solved, **evaluator_kwargs)
                transitions = [transition._replace(agent_info={'value': value.item()}) for (transition, value) in zip(transitions, values)]
                transitions[-1] = transitions[-1]._replace(done=True)
                game = [(node.state, value, action) for (node, action, _), value in zip(history, values)]
                game = [(state.get_np_array_version(), value, action) for state, value, action in game]

                return_ = sum(transition.reward for transition in transitions)
                transition_batch = data.nested_stack(transitions)
                return data.Episode(
                    transition_batch=transition_batch,
                    return_=return_,
                    solved=game_solved,
                )


def calculate_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards)+1)
    for i in np.arange(len(rewards), 0, -1):
        discounted_rewards[i-1] = gamma*discounted_rewards[i] + rewards[i-1]
    return discounted_rewards[:-1]


def game_evaluator_new(game, mode, gamma, solved, **kwargs):
    if mode == "bootstrap":
        return [node.value_acc.target() for node, _, _ in game]
    if "factual" in mode:
        rewards = np.zeros(len(game))
        if mode == "factual":  # PM: Possibly remove, kept for backward compatibility
            rewards[-1] = int(solved)
            gamma = 1
        elif mode == "factual_discount":
            rewards[-1] = int(solved)
        elif mode == "factual_hindsight":
            rewards[-1] = kwargs['hindsight_solved']
        elif mode == "factual_rewards":
            rewards = [reward for node, action, reward in game]
        else:
            raise NotImplementedError("not known mode", mode)

        return calculate_discounted_rewards(rewards, gamma)

    raise NotImplementedError("not known mode", mode)


def _softmax_sample(values_and_actions):
    # INFO: below for numerical stability,
    # see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    max_values = max([v for v, _ in values_and_actions])
    sharpening_coeff = 2.  # TODO: make it a parameter
    prob = np.array([np.exp(sharpening_coeff * v - max_values) for v, _ in values_and_actions])
    total = np.sum(prob)
    prob /= total
    idx = np.random.choice(range(len(values_and_actions)), p=prob)
    return values_and_actions[idx]


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value
