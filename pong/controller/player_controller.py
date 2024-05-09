import pygame

from .controller import Controller, MovingType, PaddlePosition

class PlayerController(Controller):
    """A controller used by player to move a paddle."""

    def __init__(self, a_paddle, position):
        """Create a player controller.
        
        Parameters
        --------------------
        a_paddle: Paddle
            a paddle to use with this controller
            
        position: PaddlePositioin
            position of paddle that controller controls"""
        
        super().__init__(a_paddle, position)

    def update(self, delta_time):
        keys = pygame.key.get_pressed()
        
        #Does player move its paddle towards upside?
        if keys[pygame.K_w] and self._position == PaddlePosition.LEFT or keys[pygame.K_UP] and self._position == PaddlePosition.RIGHT:
            self._move_paddle(MovingType.UP)
        #Does player move its paddle towards downside?
        elif keys[pygame.K_s] and self._position == PaddlePosition.LEFT or keys[pygame.K_DOWN] and self._position == PaddlePosition.RIGHT:
            self._move_paddle(MovingType.DOWN)
        #Player doesn't move its paddle.
        else:
            self._move_paddle(MovingType.NONE)
            