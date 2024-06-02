from pong.ball import Ball
from pong.paddle import Paddle

from ..train_pong_cl import TrainPongContactListener

class TrainSPPongContactListener(TrainPongContactListener):
    """A base collision system listener used for training of agent with Reinforcement Learning and self-play technique on Pong."""

    controller_2 = None          #Controller (OpponentSPController) of right paddle.

    def BeginContact(self, contact):
        super().BeginContact(contact)

        #Does paddle start contact with ball?
        if (isinstance(contact.fixtureA.userData, Paddle) and isinstance(contact.fixtureB.userData, Ball)) or (isinstance(contact.fixtureA.userData, Ball) and isinstance(contact.fixtureB.userData, Paddle)):
            paddle = contact.fixtureA.userData if isinstance(contact.fixtureA.userData, Paddle) else contact.fixtureB.userData

            #Is controller_2's paddle?
            if self.controller_2 is not None and self.current_game.paddle_2 == paddle:
                self.controller_2.n_touch += 1