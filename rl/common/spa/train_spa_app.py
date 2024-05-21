import numpy as np

from ..train_app import TrainingApp

class TrainingSPAApp(TrainingApp):
    """A base application for training of an agent with self-play method."""

    def __init__(self, training_session):
        """Create new application for training.
        
        Parameter
        --------------------
        training_session: TrainingSPASession
            a training session"""

        super().__init__(training_session)
        self._p1_infos = ""                     #Training infos of left controller.
        self._p2_infos = ""                     #Training infos of right controller.

    def _get_infos(self):
        self._p1_infos = "score = {}; avg score = {:.2f}; n. touch = {}; reward = {:.1f}; avg reward = {:.2f}".format(
                                        self._training_session.history_scores_p1[-1],
                                        np.mean(self._training_session.history_scores_p1[-100:]),
                                        self._controller_1.n_touch,
                                        self._training_session.history_rewards_p1[-1],
                                        np.mean(self._training_session.history_rewards_p1[-100:]))
        
        self._p2_infos = "score = {}; avg score = {:.2f}; n. touch = {}; reward = {:.1f}; avg reward = {:.2f}".format(
                                        self._training_session.history_scores_p2[-1],
                                        np.mean(self._training_session.history_scores_p2[-100:]),
                                        self._controller_2.n_touch,
                                        self._training_session.history_rewards_p2[-1],
                                        np.mean(self._training_session.history_rewards_p2[-100:]))

        return super()._get_infos()
    
    def _on_post_episode(self):
        super()._on_post_episode()

        self._training_session.history_rewards_p1.append(self._controller_1.total_reward)
        self._training_session.history_scores_p1.append(self._current_game.score_paddle_1 - self._current_game.score_paddle_2)

        self._training_session.history_rewards_p2.append(self._controller_2.total_reward)
        self._training_session.history_scores_p2.append(self._current_game.score_paddle_2 - self._current_game.score_paddle_1)