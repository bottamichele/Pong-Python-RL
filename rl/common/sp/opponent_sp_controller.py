from abc import abstractmethod
from pong.controller.controller import Controller, PaddlePosition, MovingType

class OpponentSPController(Controller):
    """Opponent controller used against a training agent for self-play technique."""

    def __init__(self, training_session, current_game, cl):
        """Create new controller.
        
        Parameters
        --------------------
        training_session: TrainingSPSession
            training session
            
        current game: Game
            current game of Pong
            
        cl: TrainSPPongContactListener
            a contact listener for self-play technique"""
        
        super().__init__(current_game.paddle_2, PaddlePosition.RIGHT)

        self._current_game = current_game
        self._policy = training_session.get_opponent_policy()
        self.n_touch = 0

        cl.controller_2 = self

    @abstractmethod
    def _chose_action(self):
        """Chose an action to perform.
        
        Return
        --------------------
        action: int
            action to perform"""
        
        pass

    def update(self, delta_time):
        #Chose an action to perform.
        action = self._chose_action()

        #Perform action chosen.
        self._move_paddle(MovingType(action))