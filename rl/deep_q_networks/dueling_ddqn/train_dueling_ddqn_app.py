from rl.common.train_single_agent import TrainingBotApp, BotType
from pong.controller.controller import PaddlePosition

from .dueling_ddqn_controller import DuelingDDQNTraininingBotController

class DuelingDDQNTrainingBotApp(TrainingBotApp):
    def __init__(self, training_session, opponent_type=BotType.BOT):
        super().__init__(training_session, opponent_type)

    def _create_bot_controller(self, current_game):
        return DuelingDDQNTraininingBotController(current_game.paddle_1 if self._paddle_position_to_control == PaddlePosition.LEFT else current_game.paddle_2,
                                                  self._paddle_position_to_control,
                                                  current_game,
                                                  self._training_session)

    def _get_infos(self, current_game):
        current_infos = super()._get_infos(current_game)
        other_current_infos = "states = {}; total states = {}; epsilon = {:.3f}".format(
                                                    self._training_session.states_done,
                                                    self._training_session.total_states_done,
                                                    self._training_session.epsilon)
        
        return current_infos + "; " + other_current_infos
    
    def _on_post_episode(self):
        if self._training_session.episode % 50 == 0:
            self._training_session.save_current_training_session()