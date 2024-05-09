import numpy as np

from .controller import Controller, MovingType, PaddlePosition

class BotController(Controller):
    """A controller used by CPU to move a paddle with high skill level."""

    def __init__(self, a_paddle, position, current_game):
        """Create new bot controller.
        
        Parameters
        --------------------
        a_paddle: Paddle
            a paddle to use with this controller
            
        position: PaddlePositioin
            position of paddle that controller controls
            
        current_game: Game
            current game session"""
        
        super().__init__(a_paddle, position)
        self._opponent_paddle = current_game.paddle_1 if position == PaddlePosition.RIGHT else current_game.paddle_2
        self._ball = current_game.ball
        self._field = current_game.field

    def _follow_ball(self):
        """Follow the ball."""

        if self._ball.position.y < self._paddle.position.y - self._paddle.height/2:
            self._move_paddle(MovingType.DOWN)
        elif self._ball.position.y > self._paddle.position.y + self._paddle.height/2:
            self._move_paddle(MovingType.UP)
        else:
            self._move_paddle(MovingType.NONE)

    def update(self, delta_time):
        #Is ball moving towards to me?
        if (self._ball.velocity.x > 0.0 and self._position == PaddlePosition.RIGHT) or (self._ball.velocity.x < 0.0 and self._position == PaddlePosition.LEFT):
            #Calculate time of impact between ball and a border of field.
            if self._ball.velocity.y > 0.0:
                distance_bf = (self._field.center_position.y + self._field.height/2) - (self._ball.position.y + self._ball.radius/2)
            elif self._ball.velocity.y < 0.0:
                distance_bf = (self._ball.position.y - self._ball.radius/2) - (self._field.center_position.y - self._field.height/2)
            else:
                distance_bf = np.inf
            t_impact_bf = distance_bf / abs(self._ball.velocity.y) if self._ball.velocity.y != 0 else np.inf

            #Calculate time of impact between ball and paddle towards x-axis.
            if self._position == PaddlePosition.RIGHT:      #Right paddle is mine.
                distance_bp = (self._paddle.position.x - self._paddle.width/2) - (self._ball.position.x + self._ball.radius/2)
            else:                                           #Left paddle is mine.
                distance_bp = (self._ball.position.x - self._ball.radius/2) - (self._paddle.position.x + self._paddle.width/2)
            t_impact_bp = distance_bp / abs(self._ball.velocity.x) if self._ball.velocity.x != 0 else 0.0
        
            #Does ball collides border of field before a paddle?
            if t_impact_bf != np.inf and t_impact_bf <= t_impact_bp:
                self._follow_ball()
            #Ball goes directly towards to paddle.
            else:
                delta_y_impact_bp = abs(self._ball.velocity.y) * t_impact_bp

                #Move paddle towards up to collide ball.
                if self._ball.velocity.y > 0.0 and self._paddle.position.y + self._paddle.height/2 < self._ball.position.y + delta_y_impact_bp:
                    self._move_paddle(MovingType.UP)
                #Move paddle towards down to collide ball.
                elif self._ball.velocity.y < 0.0 and self._paddle.position.y - self._paddle.height/2 > self._ball.position.y - delta_y_impact_bp:
                    self._move_paddle(MovingType.DOWN)
                else:
                    y_ball_dest = self._ball.position.y + delta_y_impact_bp if self._ball.velocity.y > 0.0 else self._ball.position.y - delta_y_impact_bp
                    
                    #Try suprpriding opponent if he has position y further than me to get one point.
                    if abs(self._paddle.position.y - self._opponent_paddle.position.y) >= 150.0:
                        #Will my paddle and ball roughly be same y value?
                        if abs(self._paddle.position.y - y_ball_dest) <= 2 * self._ball.radius:
                            self._move_paddle(MovingType.NONE)
                        #Will my paddle be below than ball?
                        elif self._paddle.position.y < y_ball_dest:
                            self._move_paddle(MovingType.UP)
                        #Will my paddle be above than ball?
                        elif self._paddle.position.y > y_ball_dest:
                            self._move_paddle(MovingType.DOWN)
                        else:
                            self._move_paddle(MovingType.NONE)
                    #Try colliding ball on (either top or bottom) corner of paddle.
                    elif abs(self._ball.position.x - self._paddle.position.x) <= 3 * self._ball.radius:
                        #Top corner of paddle is collided if final position y of ball is top.
                        if self._paddle.position.y < y_ball_dest:
                            self._move_paddle(MovingType.DOWN)
                        #Bottom corner of paddle is collided if final position y of ball is bottom.
                        elif self._paddle.position.y > y_ball_dest:
                            self._move_paddle(MovingType.UP)
                        else:
                            self._move_paddle(MovingType.NONE)
                    #Follow simply ball.
                    else:
                        self._follow_ball()
        #Ball is moving towards to opponent's paddle.
        else:
            self._follow_ball()