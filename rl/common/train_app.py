from abc import ABC, abstractmethod
from pong.game import Game

from .train_pong_cl import TrainPongContactListener
from .train_bot_controller import TrainingBotController

class TrainingApp(ABC):
    """A base application for training of agents on Pong."""

    def __init__(self, training_session):
        """Create new application to train a bot.
        
        Parameter
        --------------------
        training_session: TrainingSession
            a training session"""
        
        self._training_session = training_session
        self._time_step = 1.0 / 60.0
        self._contact_listener = None
        self._current_game = None
        self._controller_1 = None
        self._controller_2 = None

    def _create_contact_listener(self):
        """Create a new contact listener."""

        self._contact_listener = TrainPongContactListener()
    
    @abstractmethod
    def _create_controller_1(self):
        """Create new controller 1."""

        pass

    @abstractmethod
    def _create_controller_2(self):
        """Create new controller 2."""

        pass

    def _get_infos(self):
        """Print info about training.
        
        Return
        --------------------
        infos: str
            a string of all current infos training"""
        
        return "- Episode {}: {} {}".format(self._training_session.episode, 
                                            self._current_game.score_paddle_1, 
                                            self._current_game.score_paddle_2)

    def _on_post_episode(self):
        """Perform commands after an episode is ended."""
        pass

    def train(self):
        """Train bot."""

        while not self._training_session.is_ended():
            #Create new game.
            self._create_contact_listener()
            self._current_game = Game(contact_listener=self._contact_listener)
            self._current_game.start()

            #Create new controllers.
            self._create_controller_1()
            self._create_controller_2()

            if isinstance(self._controller_1, TrainingBotController):
                self._contact_listener.controller_1 = self._controller_1
            if isinstance(self._controller_2, TrainingBotController):
                self._contact_listener.controller_2 = self._controller_2

            #Game is started.
            while not self._current_game.is_ended():
                #Update current game state.
                self._current_game.update(self._time_step)
                
                #Update controller states.
                self._controller_1.update(self._time_step)
                self._controller_2.update(self._time_step)

            #Post episode.
            self._on_post_episode()

            #Print current infos.
            print(self._get_infos())

            #Next game.
            self._contact_listener = None
            self._current_game = None
            self._controller_1 = None
            self._controller_2 = None
            self._training_session.episode += 1

        self._training_session.save_model()