import numpy as np

from rl.common.sp.train_sp_app import TrainingSPApp

from .train_ddqn_sp_bot_controller import DDQNTraininingSPBotController
from .ddqn_opponent_sp_controller import DDQNOpponentSPController
from .test_ddqn_bot_controller import TestingDDQNBotController

class DDQNTrainingSPApp(TrainingSPApp):
    """Application to train a bot that uses DDQN and self-play technique."""

    def __init__(self, training_session):
        """Create new application for training of bot.
        
        Parameters
        --------------------
        training_session: DDQNTrainingSPSession
            a training session"""

        super().__init__(training_session, 50)

    def _create_controller_1(self):
        self._controller_1 = DDQNTraininingSPBotController(self._current_game, self._training_session)

    def _create_controller_2(self):
        self._controller_2 = DDQNOpponentSPController(self._training_session, self._current_game, self._contact_listener)

    def _create_test_bot_controller(self, a_game):
        return TestingDDQNBotController(self._training_session, a_game)

    def _get_infos(self):
        current_infos = super()._get_infos()
        other_current_infos = "states = {}; total states = {}".format(self._training_session.states_done, self._training_session.total_states_done)
        self._p1_infos += "; epsilon = {:.2f}".format(self._training_session.epsilon)

        return current_infos + "; " + other_current_infos + " | P1 INFOS -> " + self._p1_infos + " | P2 INFOS -> " + self._p2_infos
    
    def _on_post_episode(self):
        super()._on_post_episode()
        
        if self._training_session.episode >= 200 and self._training_session.episode % 50 == 0:
            self._training_session.save_current_training_session()