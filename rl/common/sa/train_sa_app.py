import numpy as np

from pong.controller.controller import PaddlePosition
from pong.controller.basic_bot_controller import BasicBotController
from pong.controller.bot_controller import BotController

from ..train_app import TrainingApp
from .opponent_type import OpponentType

class TrainingSAApp(TrainingApp):
    """A base application for training a single agent on Pong."""

    def __init__(self, training_session):
        """Create new application for training of an agent.
        
        Parameters
        --------------------
        training_session: TrainingSASession
            a training session
        """

        super().__init__(training_session)

    def _create_controller_2(self):
        #Create basic bot controller?
        if self._training_session.opponent_type == OpponentType.BASIC_BOT:
            self._controller_2 = BasicBotController(self._current_game.paddle_2, PaddlePosition.RIGHT, self._current_game.ball)
        #Create high skill bot controller?
        elif self._training_session.opponent_type == OpponentType.BOT:
            self._controller_2 = BotController(self._current_game.paddle_2, PaddlePosition.RIGHT, self._current_game)

    def _get_infos(self):
        current_infos = super()._get_infos()
        other_current_infos = "score = {}; avg score = {:.2f}; n. touch = {}; reward = {:.1f}; avg reward = {:.2f}".format(
                                                self._training_session.history_scores[-1],
                                                np.mean(self._training_session.history_scores[-100:]),
                                                self._controller_1.n_touch,
                                                self._training_session.history_rewards[-1],
                                                np.mean(self._training_session.history_rewards[-100:]))
        
        return current_infos + "; " + other_current_infos

    def _on_post_episode(self):
        super()._on_post_episode()

        self._training_session.history_rewards.append(self._controller_1.total_reward)
        self._training_session.history_scores.append(self._current_game.score_paddle_1 - self._current_game.score_paddle_2) 