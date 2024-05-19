from pong.game import PongGameContactListener
from pong.paddle import Paddle
from pong.ball import Ball

class TrainPongContactListener(PongGameContactListener):
    """A base collision system listener used for training of agents with Reinforcement Learning on Pong."""

    controller_1 = None          #Controller (TrainingBotController) of left paddle.
    controller_2 = None          #Controller (TrainingBotController) of right paddle

    def BeginContact(self, contact):
        super().BeginContact(contact)

        #Does paddle start contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is controller_1's paddle?
            if self.controller_1 is not None and self.controller_1.paddle == paddle:
                self.controller_1.is_colliding_ball = True
                self.controller_1.n_touch += 1
            #Is controller_2's paddle?
            elif self.controller_2 is not None and self.controller_2.paddle == paddle:
                self.controller_2.is_colliding_ball = True
                self.controller_2.n_touch += 1

    def EndContact(self, contact):
        super().EndContact(contact)

        #Does paddle end contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is controller_1's paddle?
            if self.controller_1 is not None and self.controller_1.paddle == paddle:
                self.controller_1.is_colliding_ball = False
            #Is controller_2's paddle?
            elif self.controller_2 is not None and self.controller_2.paddle == paddle:
                self.controller_2.is_colliding_ball = False