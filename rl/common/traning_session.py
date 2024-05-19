from abc import ABC, abstractmethod

class TrainingSession(ABC):
    """A session for training of agents on Pong."""

    def __init__(self):
        """Create new training session."""

        self.episode = 1            #Current episode.

    @abstractmethod
    def is_ended(self):
        """Check if training session is ended.
        
        Return
        --------------------
        is_ended: bool
            True if training session is ended, False otherwise"""
        
        pass

    @abstractmethod
    def save_model(self):
        """Save model trained on disk."""

        pass

    @abstractmethod
    def save_current_training_session(self):
        """Save current training session on disk."""

        pass

    @abstractmethod
    def load_last_training_session(self):
        """Load last training session saved on disk."""

        pass