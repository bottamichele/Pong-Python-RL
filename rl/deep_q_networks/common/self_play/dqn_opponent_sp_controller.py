import numpy as np
import torch as tc

from rl.common.sp.opponent_sp_controller import OpponentSPController
from rl.common.utils import get_full_inverse_observation_normalized

class DQNOpponentSPController(OpponentSPController):
    """A Deep Q-Networks (or any variant) controller used as opponent for training agent on Pong."""

    def _chose_action(self):
        current_observation = get_full_inverse_observation_normalized(self._current_game)
        
        #Chose action to perform.
        x = tc.Tensor( np.array([current_observation]) ).to(self._policy.model.device)
        q = self._policy.model.forward(x)
        action = tc.argmax(q).item()

        return action