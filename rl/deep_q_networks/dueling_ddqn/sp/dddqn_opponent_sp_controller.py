import torch as tc
import numpy as np

from rl.common.sp.opponent_sp_controller import OpponentSPController
from rl.common.utils import get_full_inverse_observation_normalized

class DuelingDDQNOpponentSPController(OpponentSPController):
    def __init__(self, training_session, current_game, cl):
        """Create new controller.
        
        Parameters
        --------------------
        training_session: DuelingDDQNTrainingSPSession
            training session
            
        current game: Game
            current game of Pong
            
        cl: TrainSPPongContactListener
            a contact listener for self-play technique"""
        
        super().__init__(training_session, current_game, cl)

    def _chose_action(self):
        current_observation = get_full_inverse_observation_normalized(self._current_game)
        
        #Chose action to perform.
        x = tc.Tensor( np.array([current_observation]) ).to(self._policy.model.device)
        q = self._policy.model.forward(x)
        action = tc.argmax(q).item()

        return action