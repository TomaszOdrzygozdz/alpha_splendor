from splendor.envs.mechanics.card import Card
from splendor.envs.mechanics.gems_collection import GemsCollection
from splendor.envs.mechanics.enums import GemColor
from splendor.envs.mechanics.game_settings import *


class PlayersHand:
    """A class that describes possessions of one player."""
    def __init__(self,
                 name : str = "Player") -> None:
        """Creates a hand with empty gems collections, empty set of cards, nobles and reserved cards.
        Parameters:
        _ _ _ _ _ _
        name: The name of player who has this hand (optional)."""
        self.name = name
        self.gems_possessed = GemsCollection()
        self.cards_possessed = set()
        self.cards_reserved = set()
        self.nobles_possessed = set()

    def discount(self):
        """Returns gems collection that contains the sum of profits of card possessed by the players_hand."""
        discount_dict = {gem_color : 0 for gem_color in GemColor}
        for card in self.cards_possessed:
            discount_dict[card.discount_profit] += 1
        return GemsCollection(discount_dict)

    def can_afford_card(self,
                        card: Card,
                        discount: GemsCollection = None) -> bool:
        """Returns true if players_hand can afford card"""
        if discount is None:
            discount = self.discount()
        price_after_discount = card.price % discount
        trade = [a - b for a, b in zip(price_after_discount.to_dict(), self.gems_possessed.to_dict())]
        return sum([max(a,0) for a in  trade[1:6]]) <= -trade[0]

    def min_gold_needed_to_buy_card(self,
                                    card: Card)->int:
        price_after_discount = card.price % self.discount()
        missing_gems = 0
        for gem_color in GemColor:
            if gem_color != GemColor.GOLD:
                missing_gems += max(price_after_discount.value(gem_color) - self.gems_possessed.value(gem_color),0)
        return missing_gems

    def can_reserve_card(self):
        return len(self.cards_reserved) < MAX_RESERVED_CARDS

    def number_of_my_points(self) -> int:
        return sum([card.victory_points for card in self.cards_possessed]) + sum([noble.victory_points for noble in self.nobles_possessed])

    def to_dict(self):
        return {'noble_possessed_ids' : {x.to_dict() for x in self.nobles_possessed},
                'cards_possessed_ids' : {x.to_dict() for x in self.cards_possessed},
                'cards_reserved_ids' : {x.to_dict() for x in self.cards_reserved},
                'gems_possessed' : self.gems_possessed.to_dict(),
                'name': self.name}

    def from_dict(self, vector):
        self.name  = vector['name']
        gems = vector['gems_possessed']
        self.gems_possessed = self.gems_possessed +  GemsCollection({GemColor.GOLD: gems[0], GemColor.RED: gems[1],
                                    GemColor.GREEN: gems[2], GemColor.BLUE: gems[3],
                                    GemColor.WHITE: gems[4], GemColor.BLACK: gems[5]})
