import numpy as np

from abc import abstractmethod
from collections import deque

from ..traning_session import TrainingSession


# ==================================================
# ================ GLOBAL VARIABLES ================
# ==================================================

MODEL_PATH = "./rl/models/self_play/"
"""Path of model to save on disk."""


# ==================================================
# ================== AGENT POLICY ==================
# ==================================================

class Policy:
    """A policy copied from training agent."""
    pass


# ==================================================
# =========== TRAINING SELF-PLAY SESSION ===========
# ==================================================

class TrainingSPSession(TrainingSession):
    """A session of training of an agent that uses self-play technique to learn playing on Pong."""

    def __init__(self, n_policies, copy_policy_games, change_opp_policy_games, play_last_policy_prob):
        """Create new training session.
        
        Parameters
        --------------------
        n_policies: int
            number of agent policies copied

        copy_policy_games: int
            how many games a training agent policy is copied
            
        change_opp_policy_games: int
            how many games a opponent policy is changed
            
        play_last_policy_prob: float
            probability to play against last policy copied"""
        
        super().__init__()

        self.n_policies = n_policies
        self.copy_policy_games = copy_policy_games
        self.change_opp_policy_games = change_opp_policy_games
        self.play_last_policy_prob = play_last_policy_prob        
        self.policies_copied = deque(maxlen=n_policies)                     #Policies copied from current training agent.
        self._current_opp_policy = None                                     #Current opponent's policy.
        self._rng = np.random.default_rng()

    def initialize_opponent_policy(self):
        """Initialize opponent's policy."""

        ta_policy = self.get_ta_policy()

        self._current_opp_policy = ta_policy
        self.save_policy(ta_policy)

    def save_policy(self, a_policy):
        """Save a policy copied from training agent.
        
        Parameter
        --------------------
        a_policy: Policy
            a training agent policy copied"""
        
        self.policies_copied.append(a_policy)

    @abstractmethod
    def get_ta_policy(self):
        """Return training agent's policy.
        
        Return
        --------------------
        ta_policy: Policy
            current training agent's policy"""
        
        pass

    def get_opponent_policy(self):
        """Return opponent's policy.
        
        Return
        --------------------
        opp_policy: Policy
            an opponent's policy"""
        
        #Is opponent's policy needed to change?
        if self.episode % self.change_opp_policy_games == 0:
            #Is last policy copied used?
            if self._rng.uniform() <= self.play_last_policy_prob:
                self._current_opp_policy = self.policies_copied[-1]
            else:
            #A policy is randomly chosen for opponent. 
                self._current_opp_policy = self._rng.choice(self.policies_copied)

        return self._current_opp_policy