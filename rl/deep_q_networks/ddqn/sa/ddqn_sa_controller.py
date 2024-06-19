import torch as tc

from rl.deep_q_networks.common.base_dqn_sa_controller import BaseDQNSABotController
from rl.common.sa.training_sa_session import MODEL_PATH

from .costants import MODEL_NAME
from .ddqn import DDQN

class DDQNSAController(BaseDQNSABotController):
    """A bot that uses Deep Q-Networks (DDQN) to play against a (basic or high skill) bot on Pong."""

    def _build_model(self):
        model = DDQN(self._obs_size)
        model.load_state_dict(tc.load(MODEL_PATH + MODEL_NAME + "_vs_" + self._opponent_type.name + ".pth"))

        return model
        