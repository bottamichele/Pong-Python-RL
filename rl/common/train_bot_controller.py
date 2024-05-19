from abc import abstractmethod
from pong.controller.controller import Controller, PaddlePosition, MovingType

from .utils import get_full_observation_normalized

class TrainingBotController(Controller):
    """A base class for training a bot with Reinforcement Learning on Pong."""

    def __init__(self, a_paddle, position, current_game, training_session, get_obs_fun=get_full_observation_normalized):
        """Create new bot controller to be tranined.
        
        Parameters
        --------------------
        a_paddle: Paddle
            paddle to control
            
        position: PaddlePosition
            paddle position this controller controls
            
        current_game: Game
            current game session
            
        training_session: TrainingSession
            training session
            
        get_obs_fun: callable, optional
            funtion to get a observation from a game session"""
        
        super().__init__(a_paddle, position)
        self._opponent_paddle = current_game.paddle_2 if position == PaddlePosition.LEFT else current_game.paddle_1
        self._current_game = current_game
        self._training_session = training_session
        self._get_obs_fun = get_obs_fun

        self._my_last_score = 0                 #My score of one update ago.
        self._last_opponent_score = 0           #Opponent score of one update ago.
        self.total_reward = 0                   #Total reward cumulated on this episode.
        self.n_touch = 0                        #Number of touch between my paddle and ball.
        self.is_colliding_ball = False          #True if colliding with ball, False otherwise.

        #Current infos state.
        self._current_obs = self._get_obs_fun(current_game)         #Current observation.
        self._current_next_obs = None                               #Next observation obtained performing by current action chosen.
        self._current_reward = 0                                    #Reward obtained perfoming by current action chosen.
        self._current_action = 0                                    #Current action chosen to perform.
        self._is_terminated = False                                 #Is next observation a terminal state?

        #Perform first action.
        self._chose_action()
        self._move_paddle(MovingType(self._current_action))
    
    @property
    def paddle(self):
        return self._paddle
    
    @abstractmethod
    def _chose_action(self):
        """Chose an action to perform."""
        pass

    @abstractmethod
    def _train_step():
        """Do train step."""
        pass

    def _on_post_train_step():
        """Perform commands after a train step."""
        pass
    
    def _get_reward(self):
        """Get current reward."""

        my_current_score        = self._current_game.score_paddle_1 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_2
        current_opponent_score  = self._current_game.score_paddle_2 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_1

        #Has this bot did a point?
        if my_current_score > self._my_last_score:
            self._current_reward = 1.0
        #Has this bot collided ball?
        elif self.is_colliding_ball:
            self._current_reward = 0.1
        #Has opponent did a point?
        elif current_opponent_score > self._last_opponent_score:
            self._current_reward = -1.0
        #Nothing happens.
        else:
            self._current_reward = 0.0

    def update(self, delta_time):
        #Retrieve informations about performing of current action.
        self._current_next_obs = self._get_obs_fun(self._current_game)
        self._get_reward()
        self._is_terminated = self._current_game.is_ended()

        #Train step.
        self._train_step()

        #Update informations after train step.
        self._my_last_score       = self._current_game.score_paddle_1 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_2
        self._last_opponent_score = self._current_game.score_paddle_2 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_1
        self.total_reward        += self._current_reward
        self._current_obs         = self._current_next_obs
        
        #Post train step.
        self._on_post_train_step()
        
        #Chose and perform a action.
        self._chose_action()
        self._move_paddle(MovingType(self._current_action))