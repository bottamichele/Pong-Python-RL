import pygame
from pygame.math import Vector2
from pygame.locals import *

from enum import Enum

from pong.game import Game
from pong.controller.controller import PaddlePosition
from pong.controller.player_controller import PlayerController
from pong.controller.basic_bot_controller import BasicBotController
from pong.controller.bot_controller import BotController


class ControllerType(Enum):
    """Controller type to use for paddle."""
    PLAYER = 0                      #Player controller
    BASIC_BOT = 1                   #Bot controller with basic strategy
    BOT = 2                         #Bot controller with advanced strategy
    DDQN_BOT = 3                    #Bot controller that uses DDQN
    DUELING_DDQN_BOT = 4            #Bot controller that uses Dueling DDQN


class Pong:
    """A Pong application."""

    def __init__(self, controller_1_type=ControllerType.PLAYER, controller_2_type=ControllerType.BOT):
        """Create Pong application.
        
        Parameters
        --------------------
        controller_1_type: ControllerType, optional
            controller type for left paddle
            
        contrller_2_type: ControllerType, optional
            controller type for right paddle"""

        self._is_running = False
        self._clock = pygame.time.Clock()

        #Window variables.
        self._window = None
        self._width_window = 700
        self._height_window = 550
        self._fps_limit = 60

        #Font variables.
        self._font = None
        self._color_text = (255, 255, 255)

        #Pong variables.
        self._current_game = Game()
        self._controller_1 = self._controller_factory(controller_1_type, PaddlePosition.LEFT, self._current_game)
        self._controller_2 = self._controller_factory(controller_2_type, PaddlePosition.RIGHT, self._current_game)

    def _controller_factory(self, controller_type, paddle_position, current_game):
        """Create a new controller specificed.
        
        Parameter
        --------------------
        controller_type: ControllerType
            controller type to create
            
        paddle_position: PaddlePosition
            paddle position to controller
            
        current_game: Game
            current game session"""
        
        paddle_to_control = current_game.paddle_1 if paddle_position == PaddlePosition.LEFT else current_game.paddle_2

        if controller_type == ControllerType.PLAYER:
            return PlayerController(paddle_to_control, paddle_position)
        elif controller_type == ControllerType.BASIC_BOT:
            return BasicBotController(paddle_to_control, paddle_position, current_game.ball)
        elif controller_type == ControllerType.BOT:
            return BotController(paddle_to_control, paddle_position, current_game)

    def _init(self):
        pygame.init()
        pygame.display.set_caption("Pong")

        self._is_running = True
        self._window = pygame.display.set_mode((self._width_window, self._height_window), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._font = pygame.font.Font(None, 50)

    def _shutdown(self):
        pygame.quit()
    
    def _translate_position(self, position):
        """Translate position of an object for pygame's surface coordinate.
        
        Parameter
        --------------------
        position: Vector2
            position of an object
            
        Return
        --------------------
        transl_pos: Vector2
            position translated"""
        
        new_x = position.x + self._width_window/2
        new_y = -(position.y - self._height_window/2)

        return Vector2(new_x, new_y)

    def _render(self):
        """Render phase."""

        #Fill window with colour black.
        self._window.fill("black")

        #Draw text.
        self._draw_score(self._current_game.score_paddle_1, (self._width_window // 4, 25))
        self._draw_score(self._current_game.score_paddle_2, (3 * self._width_window // 4, 25))

        #Draw borders of field.
        self._draw_border_field()

        #Draw paddles and ball on window.
        self._draw_rect(self._current_game.paddle_1.position, self._current_game.paddle_1.width, self._current_game.paddle_1.height)
        self._draw_rect(self._current_game.paddle_2.position, self._current_game.paddle_2.width, self._current_game.paddle_2.height)
        self._draw_rect(self._current_game.ball.position, self._current_game.ball.radius, self._current_game.ball.radius)

        #Put them on screen.
        pygame.display.flip()

    def _draw_score(self, score_paddle, position):
        """Draw score of a paddle on screen.
        
        Parameters
        --------------------
        score_paddle: int
            score of a paddle
            
        position: tuple
            text position on screen. It is represented as (x, y)"""
        
        #Text
        score_paddle_text = self._font.render("{}".format(score_paddle), True, self._color_text)
        
        #Text position on screen
        score_paddle_rect = score_paddle_text.get_rect()
        score_paddle_rect.center = position

        #Draw text
        self._window.blit(score_paddle_text, score_paddle_rect)

    def _draw_rect(self, position, width, height):
        """Draw a rectangle on screen.
        
        Parameters
        --------------------
        position: Vector2
            position of rectangle
            
        width: float
            width of rectangle
            
        height: float
            height of rectangle"""
        
        left_vertix_pos = self._translate_position(position + Vector2(-width/2, height/2))
        pygame.draw.rect(self._window, "white", Rect(left_vertix_pos.x, left_vertix_pos.y, width, height))

    def _draw_border_field(self, height=20):
        """Draw the borders of field.
        
        Parameter
        --------------------
        height: float
            height of the border if field"""
        
        #Top border of field.
        top_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y + self._current_game.field.height/2))
        pygame.draw.rect(self._window, "white", Rect(top_border_pos.x, top_border_pos.y - height, self._width_window, height))

        #Bottom border of field.
        bottom_border_pos = self._translate_position(Vector2(self._current_game.field.center_position.x - self._current_game.field.width/2, self._current_game.field.center_position.y - self._current_game.field.height/2))
        pygame.draw.rect(self._window, "white", Rect(bottom_border_pos.x, bottom_border_pos.y, self._width_window, height))
 
    def run(self):
        """Run Pong."""

        self._init()
        self._current_game.start()

        while self._is_running:
            #Update Pong and controllers states.
            self._controller_1.update(1.0 / self._fps_limit)
            self._controller_2.update(1.0 / self._fps_limit)
            self._current_game.update(1.0 / self._fps_limit)

            #Render.
            self._render()

            #Check if Pong game is ended
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._is_running = False

            if self._is_running:
               self._is_running = not self._current_game.is_ended()

            self._clock.tick(self._fps_limit)
            
        self._shutdown()