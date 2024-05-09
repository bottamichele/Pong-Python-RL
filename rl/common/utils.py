import numpy as np

from pygame.math import Vector2

def normalize_position(a_position, field):
    """Normalize a position of a object.
    
    Parameters
    --------------------
    a_position:
        a position of an object
        
    field: Field
        a field
        
    Return
    ---------------
    pos_normalized: Vector2
        position normalized of an object"""
    
    return Vector2((a_position.x - field.center_position.x) / (field.width/2),
                   (a_position.y - field.center_position.y) / (field.height/2))

# ==================================================
# ========= GET FULL OBSERVATION FUNCTIONS =========
# ==================================================

FULL_OBSERVATION_SIZE = 12

def get_full_observation(a_game):
    """Get a full observation of a Pong's game.
    
    Parameter
    --------------------
    a_game: Game
        a game session of Pong
        
    Return
    --------------------
    obs: ndarray
        full observation"""
    
    paddle_1_pos = a_game.paddle_1.position
    paddle_1_vel = a_game.paddle_1.velocity
    paddle_2_pos = a_game.paddle_2.position
    paddle_2_vel = a_game.paddle_2.velocity
    ball_pos = a_game.ball.position
    ball_vel = a_game.ball.velocity
    
    return np.array([paddle_1_pos.x,
                     paddle_1_pos.y,
                     paddle_1_vel.x,
                     paddle_1_vel.y,
                     paddle_2_pos.x, 
                     paddle_2_pos.y,
                     paddle_2_vel.x,
                     paddle_2_vel.y,
                     ball_pos.x,
                     ball_pos.y,
                     ball_vel.x,
                     ball_vel.y])

def get_full_observation_normalized(a_game):
        """Get a full observation normalized of a Pong's game.
        
        Parameter
        --------------------
        a_game: Game
            a game session of Pong
            
        Return
        --------------------
        obs: ndarray
            full observation normalized"""
        
        paddle_1_pos = normalize_position(a_game.paddle_1.position, a_game.field)
        paddle_1_vel = a_game.paddle_1.velocity.normalize() if a_game.paddle_1.velocity.length_squared() != 0 else Vector2()
        paddle_2_pos = normalize_position(a_game.paddle_2.position, a_game.field)
        paddle_2_vel = a_game.paddle_2.velocity.normalize() if a_game.paddle_2.velocity.length_squared() != 0 else Vector2()
        ball_pos = normalize_position(a_game.ball.position, a_game.field)
        ball_vel = a_game.ball.velocity.normalize() if a_game.ball.velocity.length_squared() != 0 else Vector2()
        
        return np.array([paddle_1_pos.x,
                         paddle_1_pos.y,
                         paddle_1_vel.x,
                         paddle_1_vel.y,
                         paddle_2_pos.x, 
                         paddle_2_pos.y,
                         paddle_2_vel.x,
                         paddle_2_vel.y,
                         ball_pos.x,
                         ball_pos.y,
                         ball_vel.x,
                         ball_vel.y])

def get_full_inverse_observation_normalized(a_game):
        """Get a full inverse observation normalized of a Pong's game.
        
        Parameter
        --------------------
        a_game: Game
            a game session of Pong
            
        Return
        --------------------
        obs: ndarray
            full inverse observation normalized"""
        
        obs = get_full_observation_normalized(a_game)

        return np.array([-obs[4], obs[5], obs[6], obs[7], -obs[0], obs[1], obs[2], obs[3], -obs[8], obs[9], -obs[10], obs[11]])