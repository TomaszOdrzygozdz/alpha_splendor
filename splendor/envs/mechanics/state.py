from typing import Set, List, Dict

from splendor.envs.mechanics.card import Card
from splendor.envs.mechanics.gems_collection import GemsCollection
from splendor.envs.mechanics.noble import Noble
from splendor.envs.mechanics.players_hand import PlayersHand
from splendor.envs.mechanics.game_settings import INITIAL_GEMS_ON_BOARD_DICT
from splendor.envs.mechanics.board import Board
from splendor.envs.cards_data.data_loader import load_all_cards
from splendor.envs.cards_data.data_loader import load_all_nobles


class State():
    """This class keeps all information about the current_state of the game."""
    type = 'state'
    def __init__(self,
                 list_of_players_hands: List = None,
                 all_cards: Set[Card] = None,
                 all_nobles: Set[Noble] = None,
                 gems_on_board: GemsCollection = None,
                 prepare_state: bool = True) -> None:

        if all_cards is None:
            all_cards = load_all_cards()
        if all_nobles is None:
            all_nobles = load_all_nobles()
        if gems_on_board is None:
            gems_on_board = GemsCollection(INITIAL_GEMS_ON_BOARD_DICT)
        if list_of_players_hands is None:
            list_of_players_hands = [PlayersHand("Player A"), PlayersHand("Player B")]

        self.list_of_players_hands = list_of_players_hands
        self.board = Board(all_cards, all_nobles, gems_on_board)
        self.active_player_id = 0

        self.who_took_last_action = None #int describing id of the player or None
        self.winner = None #int describing id of the player or None
        self.is_done = False
        self.info = {'winner' : None}
        self.steps_taken_so_far = 0

        if prepare_state:
            self.prepare()

    def prepare(self):
        self.active_player_id = 0  # index
        self.board.deck.shuffle()
        self.board.lay_cards_on_board()
        self.board.lay_nobles_on_board()

    def change_active_player(self):
        self.active_player_id = (self.active_player_id + 1)%len(self.list_of_players_hands)

    def swap_players(self):
        new_list = [None, None]
        new_list[0] = self.list_of_players_hands[1]
        new_list[1] = self.list_of_players_hands[0]
        self.list_of_players_hands = new_list

    def previous_player_id(self):
        return (self.active_player_id - 1)%len(self.list_of_players_hands)

    def active_players_hand(self):
        """Returns the hand of active player"""
        return self.list_of_players_hands[self.active_player_id]

    def other_players_hand(self):
        return self.previous_players_hand()

    def previous_players_hand(self):
        """Return the hans of the previous player"""
        return self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)]

    def set_names(self, names_tuple):
        for idx, name in enumerate(names_tuple):
            self.list_of_players_hands[idx].name = name

    def to_dict(self) -> Dict:
        return {'active_player_hand': self.active_players_hand().to_dict(),
                'other_player_hand': self.previous_players_hand().to_dict(),
                'board': self.board.to_dict(),
                'active_player_id': self.active_player_id,
                'is_done' : self.is_done,
                'winner' : self.winner,
                'steps_taken_so_far': self.steps_taken_so_far,
                'info' : self.info}
