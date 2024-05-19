import numpy as np
import torch as tc

from abc import abstractmethod
from pong.controller.controller import Controller, PaddlePosition, MovingType

from rl.common.utils import get_full_inverse_observation_normalized, FULL_OBSERVATION_SIZE

class BaseDQNSABotController(Controller):
    """A base class of an any version of Deep Q-Networks (single agent) that is used by bot controller to play on Pong."""

    def __init__(self, current_game, opponent_type, get_obs_fun=get_full_inverse_observation_normalized, obs_size=FULL_OBSERVATION_SIZE):
        """Create new controller.
        
        Parameter
        --------------------
        current_game: Game
            current session game

        opponent_type: OpponentType
            a opponent type against it was trained.
            
        get_obs_fun: callable, optional
            funtion to get a observation from a game session
            
        obs_size: int, optional
            observation size"""

        super().__init__(current_game.paddle_2, PaddlePosition.RIGHT)
        self._current_game = current_game
        self._opponent_type = opponent_type
        self._get_obs_fun = get_obs_fun
        self._obs_size = obs_size
        self._model = self._build_model()

    @abstractmethod
    def _build_model(self):
        """Build model.
        
        Return
        ------------------
        model: tc.nn.Module
            model trained"""
        
        pass
            
    def update(self, delta_time):
        current_observation = self._get_obs_fun(self._current_game)
        
        #Choose action to perform.
        x = tc.Tensor( np.array([current_observation]) ).to(self._model.device)
        q = self._model.forward(x)
        action = tc.argmax(q).item()

        #Perform action choosen.
        self._move_paddle(MovingType(action))