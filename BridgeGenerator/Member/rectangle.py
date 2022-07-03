import numpy as np

from .base import Member


class Rectangle(Member):
    def __init__(self, cfg, n, t=None, quat=None):
        """
        Create a W-beam-shape cross-section
        Args:
            b: Flange length
            h: Web length
        """
        super().__init__(cfg)
        self.b = self.shape['Flange length']
        self.h = self.shape['Web length']
        self.type = "Rectangle"

        # Initialize an empty array for 12 vertices
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        yz = np.zeros((4, 2))
        y0 = 0.5 * self.b
        z0 = 0.5 * self.h

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, -y0])
        yz[:, 1] = np.array([-z0, -z0, z0, z0])

        self.yz = yz
        self.setMember()

    def __call__(self):
        """return vertices from given parameters

        Returns:
            yz: cross-section's vertices coordinates
        """
        return self.yz
