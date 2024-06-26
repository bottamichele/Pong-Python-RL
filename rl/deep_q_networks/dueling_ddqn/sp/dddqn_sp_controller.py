import torch as tc

from pong.controller.controller import PaddlePosition
from rl.common.utils import get_full_observation_normalized, get_full_inverse_observation_normalized, FULL_OBSERVATION_SIZE
from rl.common.sp.training_sp_session import MODEL_PATH
from rl.deep_q_networks.common.base_dqn_sp_controller import BaseDQNSPBotController

from .costants import MODEL_NAME
from .dueling_ddqn import DuelingDDQN

class DuelingDDQNSPController(BaseDQNSPBotController):
    """A bot that uses Dueling Deep Q-Networks (Dueling DDQN) to play on Pong (trained with self-play technique)."""

    def __init__(self, position, current_game):
        """Create new controller.

        Parameters
        --------------------
        position: PaddlePosition
            position of paddle that controller controls

        current_game: Game
            current session game"""
        
        super().__init__(position, 
                         current_game, 
                         get_full_observation_normalized if position == PaddlePosition.LEFT else get_full_inverse_observation_normalized, 
                         FULL_OBSERVATION_SIZE)

    def _build_model(self):
        model = DuelingDDQN(self._obs_size)
        model.load_state_dict(tc.load(MODEL_PATH + MODEL_NAME + ".pth"))

        return model