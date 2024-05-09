import numpy as np

from math import sqrt
from Box2D import b2World, b2ContactListener
from pygame.math import Vector2

from .field import Field
from .paddle import Paddle
from .ball import Ball


class PongGameContactListener(b2ContactListener):
    """A base collision system listener of a Pong game."""

    current_game = None

    def BeginContact(self, contact):
        #Is collision between ball and left or right border of field?
        if (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Field)) or (isinstance(contact.fixtureA.userData, Field) and isinstance(contact.fixtureB.userData, Ball)):
            fixture_field = contact.fixtureA if isinstance(contact.fixtureA.userData, Field) else contact.fixtureB
            field = fixture_field.userData

            #Left player is assigned one point.
            if fixture_field.body == field.right_body:
                self.current_game.score_paddle_1 += 1
            #Right player is assigned one point.
            elif fixture_field.body == field.left_body:
                self.current_game.score_paddle_2 += 1

            #Reset if none touched ball.
            if fixture_field.body == field.left_body or fixture_field.body == field.right_body:
                self.current_game.is_reset_initial_state_needed = True

    def EndContact(self, contact):
        #Is collision between ball and a paddle?
        if (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)) or (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)):
            ball = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Ball) else contact.fixtureB.userData
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            vel_y_dir = sqrt(2)/2 * np.clip((ball.position.y - paddle.position.y) / paddle.height, -1, 1)
            vel_x_dir = sqrt(1 - vel_y_dir**2) * (-1 if ball.velocity.x < 0 else 1)

            ball.velocity = Ball.SPEED * Vector2(vel_x_dir, vel_y_dir)


class Game:
    """A game session of Pong."""

    def __init__(self, center_position_field=(0,0), size_field=(700, 400), size_paddle=(10, 50), radius_ball=10, score_goal=11, contact_listener=PongGameContactListener()):
        """Create a new game of Pong.
        
        Parameters
        --------------------
        center_position_field: tuple, optional
            center position of field. It is represented as (x_c, y_c) where x_c is x-axis coordinate and 
            y_c is y-axis coordinate of center position of field

        size_field: tuple, optional
            size of field. It is represented as (wf, hf) where wf is width of field and 
            hf is height of field
            
        size_paddle: tuple, optional
            size of paddle. It is represented as (wp, hp) where wp is width of paddle and
            hp is height of paddle

        radius_ball: int, optional
            radius of ball

        contact_listener: PongGameContactListener, optional
            a collision system listener
        """
    
        contact_listener.current_game = self
        self._wrldphscs = b2World(gravity=(0, 0), contactListener=contact_listener)

        self.field = Field(Vector2(center_position_field[0], center_position_field[1]), size_field[0], size_field[1], self._wrldphscs)
        self.paddle_1 = Paddle(Vector2(-0.95 * size_field[0]/2 + center_position_field[0], center_position_field[1]), size_paddle[0], size_paddle[1], self._wrldphscs)
        self.paddle_2 = Paddle(Vector2(0.95 * size_field[0]/2 + center_position_field[0], center_position_field[1]), size_paddle[0], size_paddle[1], self._wrldphscs)
        self.ball = Ball(Vector2(0, 0), radius_ball, self._wrldphscs)
        self.score_paddle_1 = 0
        self.score_paddle_2 = 0
        self._score_goal = score_goal
        self._score_done = False
        self.is_reset_initial_state_needed = False              #Used only b2ContactListener subclass.

    @property
    def score_goal(self):
        return self._score_goal
    
    def _reset_initial_state(self):
        """Reset initial state of paddles and ball."""

        #
        #Reset initial state of paddles.
        #
        self.paddle_1.position = Vector2(-0.95 * self.field.width/2 + self.field.center_position.x, self.field.center_position.y)
        self.paddle_2.position = Vector2(0.95 * self.field.width/2 + self.field.center_position.x, self.field.center_position.y)

        #
        #Reset initial state of ball.
        #
        self.ball.position = Vector2(self.field.center_position.x, self.field.center_position.y)

        # ------------------------------
        rng = np.random.default_rng()
        y_dir = rng.uniform(0.0, 0.5)
        x_dir = sqrt(1 - y_dir**2)
        vel_dir_ball = Vector2(x_dir if rng.uniform() <= 0.5 else -x_dir, y_dir if rng.uniform() <= 0.5 else -y_dir)

        self.ball.velocity = Ball.SPEED_INIT * vel_dir_ball

    def start(self):
        """Start game session."""

        self._reset_initial_state()

    def update(self, delta_time):
        """Do update step.
        
        Parameter
        --------------------
        delta_time: float
            delta time"""

        self._wrldphscs.Step(delta_time, 20, 20)

        if self.is_reset_initial_state_needed or self.field.check_ball_outside(self.ball):
            self._reset_initial_state()
            self.is_reset_initial_state_needed = False

    def is_ended(self):
        """Chech if this game session is ended.
        
        Return
        --------------------
        is_ended: bool
            True if this game session is ended, False otherwise"""
        
        return self.score_paddle_1 >= self._score_goal or self.score_paddle_2 >= self._score_goal