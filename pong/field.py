from pygame.math import Vector2
from Box2D import b2Vec2, b2FixtureDef, b2EdgeShape

from .constants import PPM

class Field:
    """A playing field of Pong."""

    def __init__(self, center_position, width, height, world_physics):
        """Create new playing field.
        
        Parameters
        --------------------
        center_position: Vector2
            center position of playing field

        width: float
            width of playing field
            
        height: float
            height of playing field
            
        world_physics: b2World
            world hub physics of Box2D"""
        
        self._center_position = center_position
        self._width = width
        self._height = height
        self._bodies = []
        
        width_fixture_def = b2FixtureDef()
        width_fixture_def.shape = b2EdgeShape(vertices=[(-0.5 * width / PPM, 0), (0.5 * width / PPM, 0)])
        width_fixture_def.density = 1
        width_fixture_def.friction = 0

        height_fixture_def = b2FixtureDef()
        height_fixture_def.shape = b2EdgeShape(vertices=[(0, -0.5 * height / PPM), (0, 0.5 * height / PPM)])
        height_fixture_def.density = 1
        height_fixture_def.friction = 0

        #Top border of field.
        self._bodies.append( world_physics.CreateStaticBody(fixtures=width_fixture_def, position=b2Vec2(center_position.x, center_position.y + height/2) / PPM) )

        #Bottom border of field.
        self._bodies.append( world_physics.CreateStaticBody(fixtures=width_fixture_def, position=b2Vec2(center_position.x, center_position.y - height/2) / PPM) )
        
        #Left border of field.
        self._bodies.append( world_physics.CreateStaticBody(fixtures=height_fixture_def, position=b2Vec2(center_position.x - width/2, center_position.y) / PPM) )

        #Right border of field.
        self._bodies.append( world_physics.CreateStaticBody(fixtures=height_fixture_def, position=b2Vec2(center_position.x + width/2, center_position.y) / PPM) )

        for body in self._bodies:
            body.userData = self
            for fixture in body.fixtures:
                fixture.userData = self

    @property
    def center_position(self):
        return Vector2(self._center_position.x, self._center_position.y)

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def left_body(self):
        return self._bodies[2]
    
    @property
    def right_body(self):
        return self._bodies[3]
    
    def check_ball_outside(self, ball):
        """Check if ball is outside of field.
        
        Parameter
        --------------------
        ball: Ball
            ball of Pong
            
        Return
        --------------------
        is_outside: bool
            True if ball is outside of field, False otherwise"""
        
        ball_pos = ball.position

        return  ball_pos.x < self._center_position.x - self._width/2 or \
                ball_pos.x > self._center_position.x + self._width/2 or \
                ball_pos.y > self._center_position.y + self._height/2 or \
                ball_pos.y < self._center_position.y - self._height/2