import random
from functools import reduce

from gym import Space

from splendor.envs.mechanics.action_space_generator import generate_all_legal_trades, generate_all_legal_buys, \
    generate_all_legal_reservations
from splendor.envs.mechanics.state import State


class SplendorActionSpace(Space):
    """A discrete space of actions in Splendor. Each action is an object of class Action."""

    def __init__(self,
                 allow_reservations: bool = True):
        super().__init__()
        self.list_of_actions = []
        self.actions_by_type = {}
        self.allow_reservations = allow_reservations

    def __iter__(self):
        return iter(self.list_of_actions)

    def __len__(self):
        return len(self.list_of_actions)

    def update(self,
               state: State) -> None:

        self.actions_by_type = {'trade': generate_all_legal_trades(state),
                                'buy': generate_all_legal_buys(state),
                                'reserve': []}
        if self.allow_reservations:
            self.actions_by_type['reserve'] : generate_all_legal_reservations(state)

        self.list_of_actions = reduce(lambda x, y: x+y, self.actions_by_type.values())

    def contains(self, x):
        """Chcecks if a given action belongs to the space"""
        return x in self.list_of_actions

    def sample(self, mode: str='from_all', list_of_preferencies=None):
        """Samples a legal action at random.
         Parameters:
          _ _ _ _ _
          mode: Determines how action will be samples. Possible values are: from all - chooses action at random from
           all legal actions, by_types - first chooses at random type of action and then chooses action of this type"""
        if len(self.list_of_actions) == 0:
            return None
        if mode == 'from_all':
            return random.choice(self.list_of_actions)
        if mode == 'by_types':
            existing_types_of_moves = [type for type in self.actions_by_type.keys() if len(self.actions_by_type[type]) > 0]
            random_type = random.choice(existing_types_of_moves)
            return random.choice(self.actions_by_type[random_type])

    def __repr__(self):
        return 'Space of Splendor legal actions, currently containing {} actions: ' \
               '{} trade gems actions, {} buy actions and {} reservation actions.'\
            .format(len(self.list_of_actions), len(self.actions_by_type['trade']),
                    len(self.actions_by_type['buy']), len(self.actions_by_type['reserve']))
    def to_dict(self):
        return [x.to_dict() for x in self.list_of_actions]