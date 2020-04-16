from collections import namedtuple

from splendor.envs.mechanics.players_hand import PlayersHand
from splendor.envs.mechanics.state import State
from splendor.envs.mechanics.state_as_dict import StateAsDict

def clone(arg):
    known_types = ('state')
    try:
        assert arg.type in known_types, f'Can clone only types: {known_types}, received {arg}'
        if arg.type == 'state':
            return StateAsDict(arg).to_state()
    except:
        raise ValueError('Argument does not have type atrribute')

PlayerStatistics = namedtuple('player', 'name pts crd nob rsv gem')

def statistics(state: State):
    def players_statistics(players_hand: PlayersHand):
        return PlayerStatistics(players_hand.name,
                                players_hand.number_of_my_points(),
                                len(players_hand.cards_possessed),
                                len(players_hand.nobles_possessed),
                                len(players_hand.cards_reserved),
                                players_hand.gems_possessed.sum())

    return tuple(players_statistics(players_hand) for players_hand in state.list_of_players_hands)