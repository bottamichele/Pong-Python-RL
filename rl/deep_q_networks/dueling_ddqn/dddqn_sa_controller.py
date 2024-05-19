import torch as tc

from rl.deep_q_networks.common.base_dqn_controller import BaseDQNSABotController

from .costants import MODEL_PATH, MODEL_NAME
from .dueling_ddqn import DuelingDDQN

class DuelingDDQNSAController(BaseDQNSABotController):
    """A bot that uses Dueling Deep Q-Networks (Dueling DDQN) to play against a (basic or high skill) bot on Pong."""

    def _build_model(self):
        model = DuelingDDQN(self._obs_size)
        model.load_state_dict(tc.load(MODEL_PATH + MODEL_NAME))

        return model
        