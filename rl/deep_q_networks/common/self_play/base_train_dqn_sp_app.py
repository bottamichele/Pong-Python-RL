from rl.common.sp.train_sp_app import TrainingSPApp

from .dqn_opponent_sp_controller import DQNOpponentSPController
from .test_dqn_sp_controller import TestDQNSPController

class BaseTrainDQNSPApp(TrainingSPApp):
    """Base application to train a bot that uses any variant of Deep Q-Networks and self-play technique."""

    def _create_controller_2(self):
        self._controller_2 = DQNOpponentSPController(self._training_session, self._current_game, self._contact_listener)

    def _create_test_bot_controller(self, a_game):
        return TestDQNSPController(self._training_session, a_game)