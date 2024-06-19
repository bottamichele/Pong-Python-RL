from ..traning_session import TrainingSession


# ==================================================
# ================ GLOBAL VARIABLES ================
# ==================================================

MODEL_PATH = "./rl/models/self_play/"
"""Path of model to save on disk."""


# ==================================================
# ========== TRAINING SINGLE AGENT SESSION =========
# ==================================================

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