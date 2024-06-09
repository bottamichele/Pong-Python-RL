import torch as tc
import numpy as np

from rl.common.sp.test_bot_controller import TestingBotController
from rl.common.utils import get_full_observation_normalized

class TestingDDQNBotController(TestingBotController):
    def _choose_action(self):
        current_observation = get_full_observation_normalized(self._current_game)
        
        #Chose action to perform.
        x = tc.Tensor( np.array([current_observation]) ).to(self._policy.model.device)
        q = self._policy.model.forward(x)
        action = tc.argmax(q).item()

        return action