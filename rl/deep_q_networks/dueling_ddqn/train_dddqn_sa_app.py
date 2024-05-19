import numpy as np

from rl.common.sa.train_sa_app import TrainingSAApp, OpponentType

from .train_dddqn_sa_bot_controller import DDDQNTraininingSABotController

class DDDQNTrainingSAApp(TrainingSAApp):
    """Application to train a bot that uses Dueling DDQN against a bot of Pong."""

    def __init__(self, training_session, opponent_type=OpponentType.BOT):
        """Create new application for training of bot controller that uses Dueling DDQN.
        
        Parameters
        --------------------
        training_session: DDDQNTrainingSASession
            a training session
            
        opponent_type: OpponentType, optional
            a opponent type against to train."""

        super().__init__(training_session, opponent_type)

    def _create_controller_1(self):
        self._controller_1 = DDDQNTraininingSABotController(self._current_game, self._training_session)

    def _get_infos(self):
        current_infos = super()._get_infos()
        other_current_infos = "states = {}; total states = {}; avg q = {:.3f}; epsilon = {:.2f}".format(
                                                    self._training_session.states_done,
                                                    self._training_session.total_states_done,
                                                    np.mean(self._training_session.history_q),
                                                    self._training_session.epsilon)
        
        return current_infos + "; " + other_current_infos
    
    def _on_post_episode(self):
        super()._on_post_episode()
        
        if self._training_session.episode % 50000000000000 == 0:
            self._training_session.save_current_training_session()