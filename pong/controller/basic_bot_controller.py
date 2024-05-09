from .controller import Controller, MovingType, PaddlePosition

class BasicBotController(Controller):
    """A controller used by CPU to move a paddle with a basic strategy."""

    def __init__(self, a_paddle, position, ball):
        """Create a basic bot controller.
        
        Parameters
        --------------------
        a_paddle: Paddle
            a paddle to use with this controller
            
        position: PaddlePositioin
            position of paddle that controller controls
            
        ball: Ball
            ball of Pong"""
        
        super().__init__(a_paddle, position)
        self._ball = ball

    def update(self, delta_time):
        #Is ball moving towards to me?
        if (self._ball.velocity.x > 0.0 and self._position == PaddlePosition.RIGHT) or (self._ball.velocity.x < 0.0 and self._position == PaddlePosition.LEFT):
            if self._ball.position.y < self._paddle.position.y - self._paddle.height/2:
                self._move_paddle(MovingType.DOWN)
            elif self._ball.position.y > self._paddle.position.y + self._paddle.height/2:
                self._move_paddle(MovingType.UP)
            else:
                self._move_paddle(MovingType.NONE)
        else:
            self._move_paddle(MovingType.NONE)