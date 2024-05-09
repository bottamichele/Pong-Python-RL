class BoxCollider:
    """A bounding box collider."""

    def __init__(self, position, width, height):
        """Create box collider.
        
        Parameters
        --------------------
        position: Vector2
            top left vertix position of bounding box collider

        width: float
            width of bounding box collider

        height: float
            height of bounding box collider"""

        self.position = position
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    def check_collision_point(self, p_pos):
        """Check if this collides a point at position p_pos.
        
        Parameter
        --------------------
        p_pos: Vector2
            position of point

        Return
        --------------------
        is_collided: bool
            True if this collides the point, False otherwise"""

        return  self.position.x <= p_pos.x and p_pos.x <= self.position.x + self._width   and \
                self.position.y <= p_pos.y and p_pos.y <= self.position.y + self._height

    def check_collision(self, other):
        """Check this collides with other bounding box collider.
        
        Parameter
        --------------------
        other: BoxCollider
            an other bounding box collider
            
        Return
        --------------------
        is_collided: bool
            True if this and other was collided, False otherwise"""

        return  not(self.position.x + self._width < other.position.x or self.position.x > other.position.x + other.width) and \
                not(self.position.y + self._height < other.position.y or self.position.y > other.position.y + other.height)