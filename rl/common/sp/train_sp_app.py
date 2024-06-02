from ..train_app import TrainingApp

from .train_sp_pong_cl import TrainSPPongContactListener

class TrainingSPApp(TrainingApp):
    """A base application for training of an agent that uses self-play technique to train playing on Pong."""

    def __init__(self, training_session):
        """Create new application for training.
        
        Parameter
        --------------------
        training_session: TrainingSPSession
            a training session"""

        super().__init__(training_session)
        self._p1_infos = ""                     #Training infos of left controller.
        self._p2_infos = ""                     #Infos of right controller.

    def _create_contact_listener(self):
        self._contact_listener = TrainSPPongContactListener()

    def _get_infos(self):
        self._p1_infos = "n. touch = {}; reward = {:.1f}".format(self._controller_1.n_touch, self._controller_1.total_reward)
        self._p2_infos = "n. touch = {}".format(self._controller_2.n_touch)

        return super()._get_infos()
    
    def _on_post_episode(self):
        super()._on_post_episode()

        if self._training_session.episode % self._training_session.copy_policy_games == 0:
            self._training_session.save_policy(self._training_session.get_ta_policy())