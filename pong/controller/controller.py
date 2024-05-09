from abc import ABC, abstractmethod
from enum import Enum

from pygame.math import Vector2

from ..paddle import Paddle

class MovingType(Enum):
    """Moving type can be choosen by a controller."""
    NONE = 0            #Paddle is not moved.
    UP = 1              #Paddle is moved towards up.
    DOWN = 2            #Paddle is moved towards down.


class PaddlePosition(Enum):
    """Position of paddle that player controls"""
    LEFT = 0        #Player controls left paddle.
    RIGHT = 1       #Player controls right paddle.


class Controller(ABC):
    """A controller to move a paddle."""

    def __init__(self, a_paddle, position):
        """Create controller for a paddle.
        
        Parameters
        --------------------
        a_paddle: Paddle
            a paddle to control with this controller
            
        position: PaddlePositioin
            position of paddle that controller controls"""
        
        self._paddle = a_paddle
        self._position = position

    @abstractmethod
    def update(self, delta_time):
        """Update controller.
        
        Parameter
        --------------------
        delta_time: float
            delta time"""
        
        pass

    def _move_paddle(self, moving_type):
        """Move paddle."""

        if moving_type == MovingType.NONE:
            self._paddle.velocity = Vector2(0.0, 0.0)
        elif moving_type == MovingType.UP:
            self._paddle.velocity = Vector2(0.0, Paddle.SPEED)
        elif moving_type == MovingType.DOWN:
            self._paddle.velocity = Vector2(0.0, -Paddle.SPEED)