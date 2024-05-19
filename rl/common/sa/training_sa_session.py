from ..traning_session import TrainingSession

class TrainingSASession(TrainingSession):
    """A session for traning of a single agent on Pong."""

    def __init__(self):
        super().__init__()

        self.history_rewards = []           #History of previous match rewards.
        self.history_scores = []            #History of previous match scores.