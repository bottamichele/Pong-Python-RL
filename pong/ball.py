from pygame.math import Vector2
from Box2D import b2Vec2, b2PolygonShape

from .constants import PPM

class Ball:
    """A ball of Pong."""

    SPEED = 600
    SPEED_INIT = 300

    def __init__(self, position, radius, world_physics):
        """Create new ball.
        
        Parameters
        --------------------
        position: Vector2
            initial position of ball
            
        radius: float
            radius of ball
            
        world_physics: b2World
            world hub physics of Box2D"""        

        self._radius = radius
        self._rigid_body = world_physics.CreateDynamicBody(position=(position.x / PPM, position.y / PPM),
                                                           linearVelocity=(0, 0),
                                                           fixedRotation=True,
                                                           bullet=True)
        self._fixture = self._rigid_body.CreateFixture(shape=b2PolygonShape(box=(0.5 * radius / PPM, 0.5 * radius / PPM)),
                                                       density=1,
                                                       restitution=1,
                                                       friction=0)
        self._rigid_body.mass = 10**-4

        self._rigid_body.userData = self
        self._fixture.userData = self

    @property
    def position(self):
        return PPM * Vector2(self._rigid_body.position.x, self._rigid_body.position.y)
    
    @position.setter
    def position(self, new_pos):
        if not isinstance(new_pos, Vector2):
            raise TypeError("expected Vector2 for position")
        
        self._rigid_body.position = b2Vec2(new_pos.x, new_pos.y) / PPM

    @property
    def velocity(self):
        return PPM * Vector2(self._rigid_body.linearVelocity.x, self._rigid_body.linearVelocity.y)
    
    @velocity.setter
    def velocity(self, new_vel):
        if not isinstance(new_vel, Vector2):
            raise TypeError("expected Vector2 for velocity")
        
        self._rigid_body.linearVelocity = b2Vec2(new_vel.x, new_vel.y) / PPM

    @property
    def rigid_body(self):
        return self._rigid_body
    
    @property
    def radius(self):
        return self._radius