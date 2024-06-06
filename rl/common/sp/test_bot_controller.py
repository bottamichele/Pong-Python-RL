from abc import abstractmethod
from pong.controller.controller import Controller, PaddlePosition, MovingType

class TestingBotController(Controller):
    """A controller to test training policy."""

    def __init__(self, training_session, a_game):
        """Create new controller.
        
        Parameters
        --------------------
        training_session: TrainingSPSession
            a training session
            
        a_game: Game
            a game session"""
        
        super().__init__(a_game.paddle_1, PaddlePosition.LEFT)
        
        self._policy = training_session.get_ta_policy()
        self._current_game = a_game

    @abstractmethod
    def _choose_action(self):
        """Choose an action to perform.
        
        Return
        --------------------
        action: int
            action to perform"""
        
        pass

    def update(self, delta_time):
        #Chose an action to perform.
        action = self._choose_action()

        #Perform action chosen.
        self._move_paddle(MovingType(action))