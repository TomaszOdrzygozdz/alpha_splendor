from typing import List

from splendor.envs.mechanics.action import Action, ActionTradeGems, ActionBuyCard, ActionReserveCard
from splendor.envs.mechanics.game_settings import USE_FAST_ACTION_GENERATOR
from splendor.envs.mechanics.state import State


if USE_FAST_ACTION_GENERATOR:
    from splendor.envs.mechanics.action_space_generator_fast import generate_all_legal_trades_fast
    from splendor.envs.mechanics.action_space_generator_fast import generate_all_legal_buys_fast
    from splendor.envs.mechanics.action_space_generator_fast import generate_all_legal_reservations_fast

    generate_all_legal_trades = generate_all_legal_trades_fast
    generate_all_legal_buys = generate_all_legal_buys_fast
    generate_all_legal_reservations = generate_all_legal_reservations_fast

else:
    from splendor.envs.mechanics.action_space_generator_classic import generate_all_legal_trades_classic
    from splendor.envs.mechanics.action_space_generator_classic import generate_all_legal_buys_classic
    from splendor.envs.mechanics.action_space_generator_classic import generate_all_legal_reservations_classic

    generate_all_legal_trades = generate_all_legal_trades_classic
    generate_all_legal_buys = generate_all_legal_buys_classic
    generate_all_legal_reservations = generate_all_legal_reservations_classic

def generate_all_legal_actions(state: State) -> List[Action]:
    """Generates list of all possible actions of an active player in a given current_state."""
    return generate_all_legal_trades(state) + generate_all_legal_buys(state) + generate_all_legal_reservations(
        state)
