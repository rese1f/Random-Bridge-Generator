import numpy as np

from .base import CrossSection


class wBeam(CrossSection):
    def __init__(self, cfg):
        """
        Create a W-beam-shape cross-section
        Args:
            b: Flange length
            h: Web length
            tf: Flange thickness
            tw: Web thickness
        """
        super().__init__(cfg)
        self.b = self.shape['Flange length']
        self.h = self.shape['Web length']
        self.tf = self.shape['Flange thickness']
        self.tw = self.shape['Web thickness']
    
    def __call__(self):
        """return vertices from given parameters

        Returns:
            yz: cross-section's vertices coordinates
        """
        # Initialize an empty array for 12 vertices
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        yz = np.zeros((12, 2))
        y0, y1 = 0.5 * self.b, 0.5 * self.tw
        z0, z1 = 0.5 * self.h, 0.5 * self.h - self.tf

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, y1, y1, y0, y0, -y0, -y0, -y1, -y1, -y0])
        yz[:, 1] = np.array([-z0, -z0, -z1, -z1, z1, z1, z0, z0, z1, z1, -z1, -z1])

        return yz