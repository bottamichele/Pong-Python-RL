import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from pong.game import Game, PongGameContactListener
from pong.controller.controller import Controller, MovingType, PaddlePosition
from pong.controller.basic_bot_controller import BasicBotController
from pong.controller.bot_controller import BotController
from pong.ball import Ball
from pong.paddle import Paddle

from .utils import get_full_observation_normalized


# ==================================================
# ================ TRAINING SESSION ================
# ==================================================

class TrainingSession(ABC):
    """A session for training a bot at Pong."""

    def __init__(self):
        """Create new training session."""

        self.episode = 1                    #Current episode.
        self.total_reward = 0               #Total reward cumulated from last episode.
        self.history_rewards = []           #History of previous match rewards.
        self.history_scores = []            #History of previous match scores.

    @abstractmethod
    def is_ended(self):
        """Check if training session is ended.
        
        Return
        --------------------
        is_ended: bool
            True if training session is ended, False otherwise"""
        
        pass

    @abstractmethod
    def save_model(self):
        """Save model trained on disk."""

        pass

    @abstractmethod
    def save_current_training_session(self):
        """Save current training session on disk."""

        pass

    @abstractmethod
    def load_last_training_session(self):
        """Load last training session saved on disk."""

        pass


# ==================================================
# ============= TRAINING BOT CONTROLLER ============
# ==================================================

class TrainingBotController(Controller):
    """A base class for training a bot with Reinforcement Learning on Pong (single agent)."""

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
        self._rng = np.random.default_rng()

        self._my_last_score = 0
        self._last_opponent_score = 0
        self._training_session.total_reward = 0
        self.is_colliding_ball = False

        #Current infos state.
        self._current_obs = self._get_obs_fun(current_game)
        self._current_next_obs = None
        self._current_reward = 0
        self._current_action = 0
        self._is_terminated = False

        #Perform first action.
        self._choose_action()
        self._move_paddle(self._current_action)
    
    @property
    def paddle(self):
        return self._paddle
    
    @abstractmethod
    def _choose_action(self):
        """Choose an action to perform."""
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
        #Nothing.
        else:
            self._current_reward = 0.0

    def update(self, delta_time):
        self._current_next_obs = self._get_obs_fun(self._current_game)
        self._get_reward()
        self._is_terminated = self._current_game.is_ended()
        self._train_step()

        self._my_last_score                  = self._current_game.score_paddle_1 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_2
        self._last_opponent_score            = self._current_game.score_paddle_2 if self._position == PaddlePosition.LEFT else self._current_game.score_paddle_1
        self._training_session.total_reward += self._current_reward
        self._current_obs                    = self._current_next_obs
        self._on_post_train_step()
        
        self._choose_action()
        self._move_paddle(MovingType(self._current_action))


# ==================================================
# ========== TRAINING BOT CONTACT LISTENER =========
# ==================================================

class TrainingBotContactListener(PongGameContactListener):
    """A base collision system listener used for training of a bot with Reinforcement Learning."""

    bot_controller = None               #Controller used to train bot.
    opponent_controller = None          #Controller used by opponent.

    def BeginContact(self, contact):
        super().BeginContact(contact)

        #Does paddle start contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            if self.bot_controller.paddle == paddle:
                self.bot_controller.is_colliding_ball = True

    def EndContact(self, contact):
        super().EndContact(contact)

        #Does paddle end contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            if self.bot_controller.paddle == paddle:
                self.bot_controller.is_colliding_ball = False


# ==================================================
# ============ TRAINING BOT APPLICATION ============
# ==================================================

class BotType(Enum):
    """Type of bot to play against."""
    BASIC_BOT = 0                   #Bot with basic strategy
    BOT = 1                         #Bot with advanced strategy

class TrainingBotApp(ABC):
    def __init__(self, training_session, opponent_type=BotType.BOT):
        """Create new application to train a bot.
        
        Parameter
        --------------------
        training_session: TrainingSession
            a training session for application
            
        opponent_type: BotType, optional
            type of bot to use for training"""
        
        self._paddle_position_to_control = PaddlePosition.LEFT
        self._training_session           = training_session
        self._opponent_type              = opponent_type
        self._time_step                  = 1.0 / 60.0

    def _create_contact_listener(self):
        """Create a new contact listener.
        
        Return
        --------------------
        cl: TrainingBotContactListener
            System collision listener"""
        
        return TrainingBotContactListener()
    
    @abstractmethod
    def _create_bot_controller(self, current_game):
        """Create new bot controller to train.

        Parameter
        --------------------
        current_game: Game
            current game session
        
        Return
        --------------------
        bot_controller: TrainingBotController
            New bot controller to train"""
        
        pass

    def _create_opponent_controller(self, current_game):
        """Create new opponent controller.
        
        Parameter
        --------------------
        current_game: Game
            current game session
            
        Return
        --------------------
        opponent_controller: Controller
            New opponent controller"""
        
        paddle_to_control = current_game.paddle_2 if self._paddle_position_to_control == PaddlePosition.LEFT else current_game.paddle_1
        position_paddle_to_control = PaddlePosition.RIGHT if self._paddle_position_to_control == PaddlePosition.LEFT else PaddlePosition.LEFT
        
        if self._opponent_type == BotType.BASIC_BOT:
            return BasicBotController(paddle_to_control, position_paddle_to_control, current_game.ball)
        elif self._opponent_type == BotType.BOT:
            return BotController(paddle_to_control, position_paddle_to_control, current_game)

    def _get_infos(self, current_game):
        """Print info about training.
        
        Parameter
        --------------------
        current_game: Game
            current game session
            
        Return
        --------------------
        infos: str
            a string of all current infos"""
        
        return "- Episode {}: {} {}; score = {}; avg score = {:.2f}; n. touch = {:.0f}; reward = {:.1f}; avg reward = {:.2f}".format(
                                self._training_session.episode, 
                                current_game.score_paddle_1, 
                                current_game.score_paddle_2, 
                                self._training_session.history_scores[-1], 
                                np.mean(self._training_session.history_scores[-100:]),
                                self._count_touch(current_game),
                                self._training_session.history_rewards[-1],
                                np.mean(self._training_session.history_rewards[-100:]))

    def _on_post_episode(self):
        """Perform commands after an episode is ended."""
        pass

    def _count_touch(self, current_game):
        """Count how many times current bot training touched ball.
        
        Parameter
        --------------------
        current_game: Game
            current game session
            
        n_touches: int
            number of touch done on ball"""
        
        my_final_score       = current_game.score_paddle_1 if self._paddle_position_to_control == PaddlePosition.LEFT else current_game.score_paddle_2
        opponent_final_score = current_game.score_paddle_2 if self._paddle_position_to_control == PaddlePosition.LEFT else current_game.score_paddle_1
        final_total_reward   = self._training_session.total_reward

        return (final_total_reward + opponent_final_score - my_final_score) / 0.1

    def train(self):
        """Train bot."""

        while not self._training_session.is_ended():
            #Create new game.
            contact_listener = self._create_contact_listener()
            current_game = Game(contact_listener=contact_listener)
            current_game.start()

            #Create new controllers.
            bot_controller      = self._create_bot_controller(current_game)
            opponent_controller = self._create_opponent_controller(current_game)
            contact_listener.bot_controller      = bot_controller
            contact_listener.opponent_controller = opponent_controller

            #Game is started.
            while not current_game.is_ended():
                #Update current game state.
                current_game.update(self._time_step)
                
                #Update controller states.
                bot_controller.update(self._time_step)               #Train step for bot.
                opponent_controller.update(self._time_step)

            #Update history.
            self._training_session.history_scores.append((current_game.score_paddle_1 - current_game.score_paddle_2) * (-1 if self._paddle_position_to_control == PaddlePosition.RIGHT else 1))
            self._training_session.history_rewards.append(self._training_session.total_reward)

            #Post episode.
            self._on_post_episode()

            #Print current infos.
            print(self._get_infos(current_game))

            #Next game.
            self._training_session.episode += 1

        self._training_session.save_model()