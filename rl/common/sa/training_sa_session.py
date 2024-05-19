from ..traning_session import TrainingSession

class TrainingSASession(TrainingSession):
    """A session for traning of a single agent on Pong."""

    def __init__(self, opponent_type):
        """Create new training session.
        
        opponent_type: OpponentType
            a opponent type against to train."""
        
        super().__init__()

        self.opponent_type = opponent_type
        self.history_rewards = []                   #History of previous match rewards.
        self.history_scores = []                    #History of previous match scores.