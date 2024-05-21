import numpy as np
import torch as tc

from abc import abstractmethod
from pong.controller.controller import Controller, PaddlePosition, MovingType

class BaseDQNSPABotController(Controller):
    """A base class of an any version of Deep Q-Networks (agent trained with self-play method) that is used by bot controller to play on Pong."""

    def __init__(self, position, current_game, get_obs_fun, obs_size):
        """Create new controller.
        
        Parameter
        --------------------
        position: PaddlePosition
            position of paddle that controller controls 

        current_game: Game
            current session game
            
        get_obs_fun: callable
            funtion to get a observation from a game session
            
        obs_size: int
            observation size"""

        super().__init__(current_game.paddle_1 if position == PaddlePosition.LEFT else current_game.paddle_2, 
                         position)

        self._current_game = current_game
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