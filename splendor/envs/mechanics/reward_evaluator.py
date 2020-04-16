import gin
from abc import abstractmethod

class RewardEvaluator:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, action, state, is_done, who_took_last_action):
        raise NotImplementedError

class OnlyVictory(RewardEvaluator):
    def __init__(self):
        super().__init__('OnlyVictory')

    def evaluate(self, action, state, is_done, who_took_last_action):
        if not is_done:
            return 0
        else:
            if action is not None:
                return 1
            else:
                return 0