from abc import abstractmethod

from pong.game import Game
from pong.controller.controller import PaddlePosition
from pong.controller.basic_bot_controller import BasicBotController
from pong.controller.bot_controller import BotController

from ..train_app import TrainingApp
from .train_sp_pong_cl import TrainSPPongContactListener

class TrainingSPApp(TrainingApp):
    """A base application for training of an agent that uses self-play technique to train playing on Pong."""

    def __init__(self, training_session, test_performace_games):
        """Create new application for training.
        
        Parameter
        --------------------
        training_session: TrainingSPSession
            a training session
            
        test_performance_games: int
            how many games performance of training agent is tested"""

        super().__init__(training_session)

        self._test_performance_games = test_performace_games
        self._p1_infos = ""                     #Training infos of left controller.
        self._p2_infos = ""                     #Infos of right controller.

    def _create_contact_listener(self):
        self._contact_listener = TrainSPPongContactListener()

    @abstractmethod
    def _create_test_bot_controller(self, a_game):
        """Create bot controller to test its policy.
        
        Return
        --------------------
        tb_controller: TestingBotController
            controller for testing"""
        
        pass

    def _get_infos(self):
        self._p1_infos = "n. touch = {}; reward = {:.1f}".format(self._controller_1.n_touch, self._controller_1.total_reward)
        self._p2_infos = "n. touch = {}".format(self._controller_2.n_touch)

        return super()._get_infos()
    
    def _test_training_agent(self):
        """Test performance of training agent."""

        print("------------------------------------------------------------")
        print("- Test Phase")
        
        n_test_games = 2
        for test_game in range(1, n_test_games+1):
            #A new game session is created.
            game = Game()

            #Create controllers.
            test_bot_controller = self._create_test_bot_controller(game)
            if test_game == 1:     
                opponent_controller = BasicBotController(game.paddle_2, PaddlePosition.RIGHT, game.ball)
            else:
                opponent_controller = BotController(game.paddle_2, PaddlePosition.RIGHT, game)

            #Game is started.
            game.start()
            while not game.is_ended():            
                #Update controller states.
                test_bot_controller.update(self._time_step)
                opponent_controller.update(self._time_step)

                #Update current game state.
                game.update(self._time_step)

            print("- Score: TRAINING_AGENT = {}; {} = {}".format(game.score_paddle_1,
                                                                 "BASIC_BOT" if test_game == 1 else "BOT",
                                                                 game.score_paddle_2))

        print("------------------------------------------------------------")
    
    def _on_post_episode(self):
        super()._on_post_episode()

        if self._training_session.episode % self._test_performance_games == 0:
            self._test_training_agent()

        if self._training_session.episode % self._training_session.copy_policy_games == 0:
            self._training_session.save_policy(self._training_session.get_ta_policy())    