from ..traning_session import TrainingSession

class TrainingSPASession(TrainingSession):
    """A session of training of an agent on Pong with self-play method."""

    def __init__(self):
        super().__init__()

        self.history_rewards_p1 = []                    #History of previous match rewards of left controller.
        self.history_scores_p1 = []                     #History of previous match scores of left controller.
        self.history_rewards_p2 = []                    #History of previous match rewards of right controller.
        self.history_scores_p2 = []                     #History of previous match scores of right controller.