# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:14:17 2022

@author: naraz, Jiabao
"""

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class CrossSection:
    """
    define cross section in the local coordinate system
    x is the direction perpendicular to the cross-section
    cross-sections are defined in yz plane
    """

    def __init__(self):
        """
        Member variables:
            C: Accumulative cross-section
            yz: Current cross-section's vertices
        """
        self.C = None
        self.yz = None

    def A_shape_column(self, w_2, h_1, h_2, h_3, h_4, h_5, k, bridge_thick):
        """
        见示意图

        """
        w_1 = (h_1+h_2+h_3)/k
        yz1 = np.array([
            [0, (h_1+h_2+h_3+h_4+h_5)],
            [0, (h_1+h_2+h_3)],
            [h_3/k, (h_1+h_2)],
            [0, (h_1+h_2)],
            [0, h_1],
            [(h_2+h_3)/k, h_1],            
            [w_1, 0],
            [w_1+w_2, 0],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4)],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4+h_5)]
        ])

        yz2 = np.zeros(yz1.shape)
        yz2[:, 0] = -yz1[:, 0]
        yz2[:, 1] = yz1[:, 1]
        yz = np.concatenate((yz1,yz2), 0)
        bridge_max_width = (h_3-bridge_thick)/k * 2
        self.yz = yz
        self.add_current_section()
        return yz, bridge_max_width, h_5 

    # def rectangle(self, b, h):
    #     """
    #     Create a rectangle cross-section.
    #     Args:
    #         b: Flange length
    #         h: Web length
    #
    #     Returns:
    #         yz: rectangle cross-section's vertices coordinates
    #     """
    #     # Initialize an empty array for four vertices
    #     # Possible absolute value(s) for y
    #     # Possible absolute value(s) for z
    #     yz = np.zeros((4, 2))
    #     y0 = 0.5 * b
    #     z0 = 0.5 * h
    #
    #     # The rows in yz are the coordinates (y,z).
    #     # Begin from the left-bottom corner and add other points counterclockwise
    #     yz[:, 0] = np.array([-y0, y0, y0, -y0])
    #     yz[:, 1] = np.array([-z0, -z0, z0, z0])
    #
    #     # Update self.yz
    #     # Update self.C
    #     self.yz = yz
    #     self.add_current_section()
    #     return yz

    def w_beam(self, b, h, tf, tw):
        """
        Create a W-beam-shape cross-section
        Args:
            b: Flange length
            h: Web length
            tf: Flange thickness
            tw: Web thickness

        Returns:
            yz: W-beam cross-section's vertices coordinates
        """
        # Initialize an empty array for 12 vertices
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        yz = np.zeros((12, 2))
        y0, y1 = 0.5 * b, 0.5 * tw
        z0, z1 = 0.5 * h, 0.5 * h - tf

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, y1, y1, y0, y0, -y0, -y0, -y1, -y1, -y0])
        yz[:, 1] = np.array([-z0, -z0, -z1, -z1, z1, z1, z0, z0, z1, z1, -z1, -z1])

        # Update self.yz
        # Update self.C
        self.yz = yz
        self.add_current_section()
        return yz

    def l_beam(self, b, h, tf, tw):
        """
        Create a L-shape cross-section.
        Args:
            b: Flange length
            h: Web length
            tf: Flange thickness
            tw: Web thickness

        Returns:
            yz: L-beam cross-section's vertices coordinates
        """
        # Initialize an empty array for 12 vertices
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        # Assume the initial origin is at the middle of L-beam
        yz = np.zeros((6, 2))
        y0, y1 = 0.5 * b, 0.5 * b - tw
        z0, z1 = 0.5 * h, 0.5 * h - tf

        # Calculate the location of center of mass
        A1, A2 = h*tw, (b-tw)*tf
        c1, c2 = np.array([-y0+tw/2, 0]), np.array([tw/2, -z0+tf/2])
        c = (A1*c1+A2*c2)/(A1+A2)

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, -y1, -y1, -y0]) - c[0]
        yz[:, 1] = np.array([-z0, -z0, -z1, -z1, z0, z0]) - c[1]
        self.yz = yz
        self.add_current_section()
        return yz

    def wt_beam(self, b, h, tf, tw):
        """
        Create a T-shape cross-section.
        Args:
            b: Flange length
            h: Web length
            tf: Flange thickness
            tw: Web thickness

        Returns:
            yz: WT-beam cross-section's vertices coordinates
        """
        # Initialize an empty array for 8 vertices
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        # Assume the initial origin is at the middle of WT-beam
        yz = np.zeros((8, 2))
        y0, y1 = 0.5 * b, 0.5 * tw
        z0, z1 = 0.5 * h, 0.5 * h - tf

        # Calculate the location of center of mass
        A1, A2 = b*tf, (h-tf)*tw
        c_z1, c_z2 = -z0+tf/2, tw/2
        c_z = (A1*c_z1 + A2*c_z2)/(A1+A2)

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, y1, y1, -y1, -y1, -y0])
        yz[:, 1] = np.array([-z0, -z0, -z1, -z1, z0, z0, -z1, -z1]) - c_z
        self.yz = yz
        self.add_current_section()
        return yz

    def double_l_beam(self, b, h, tf, tw, d):
        """
        Create a 2L-beams cross-section.
        Args:
            b: Flange length
            h: Web length
            tf: Flange thickness
            tw: Web thickness
            d: Distance between two L-beams

        Returns:
            yz: 2L-beam cross-section's vertices coordinates
        """
        # Initialize an empty array for 12 vertices, 6 for each L-beam
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        # Assume the initial origin is at the middle of 2L-beam
        yz = np.zeros((12, 2))
        y0, y1, y2 = 0.5 * d, 0.5 * d + b, 0.5 * d + tw
        z0, z1 = 0.5 * h, 0.5 * h - tf

        # Calculate the location of center of mass
        A1, A2 = h * tw, (b - tw) * tf
        c_z1, c_z2 = 0, -z0 + tf / 2
        c_z = (A1 * c_z1 + A2 * c_z2) / (A1 + A2)

        # The rows in yz are the coordinates (y,z)
        # The first half begins from the left-bottom corner and add other points counterclockwise
        # The second half and the first half are symmetric about the z axis
        yz[:, 0] = np.array([-y1, -y0, -y0, -y2, -y2, -y1, y1, y0, y0, y2, y2, y1])
        yz[:, 1] = np.array([-z0, -z0, z0, z0, -z1, -z1, -z0, -z0, z0, z0, -z1, -z1]) - c_z
        self.yz = yz
        self.add_current_section()
        return yz

    def hss_beam(self, b, h, w):
        """
        Create a square hollow structural(HSS) beam cross-section.
        Args:
            b: Flange length
            h: Web length
            w: Width

        Returns:
            yz: HSS-beam cross-section's vertices coordinates
        """
        # Initialize an empty array for 8 vertices, 4 for inside and outside rectangles
        # Possible absolute value(s) for y
        # Possible absolute value(s) for z
        yz = np.zeros((8, 2))
        y0, y1 = b/2, b/2 - w
        z0, z1 = h/2, h/2 - w

        # The rows in yz are the coordinates (y,z).
        # Begin from the left-bottom corner and add other points counterclockwise
        yz[:, 0] = np.array([-y0, y0, y0, -y0, -y1, y1, y1, -y1])
        yz[:, 1] = np.array([-z0, -z0, z0, z0, -z1, -z1, z1, z1])
        self.yz = yz
        self.add_current_section()
        return yz

    def pipe(self, r, w=0, inner=True, n=100):
        """
        Create a pipe cross-section.
        Args:
            r: Outer radius
            w: Width
            inner: Boolean
                Inner==True, create mesh for the inner side of the cross-section
                Inner==False, create outside only
            n: Number of points on circle

        Returns:
            yz: Pipe cross-section's vertices coordinates
        """
        # Initialize angles between 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, n)

        if inner:
            yz = np.zeros((2 * n, 2))
            # Create the ring's coordinates
            yz[:, 0] = np.hstack([r * np.cos(theta), (r - w) * np.cos(theta)])
            yz[:, 1] = np.hstack([r * np.sin(theta), (r - w) * np.sin(theta)])
        else:
            yz = np.zeros((n, 2))
            # Create the circle's coordinates
            yz[:, 0] = r * np.cos(theta)
            yz[:, 1] = r * np.sin(theta)

        self.yz = yz
        self.add_current_section()
        return yz

    def add_current_section(self):
        """
        Update self.C by adding the current section
        """
        # Reshape yz from 2D to 3D.
        yz = self.yz.reshape((1, self.yz.shape[0], self.yz.shape[1]))
        if self.C is None:
            self.C = yz
        else:
            # Add the current section to C along axis = 0.
            self.C = np.concatenate((self.C, yz), 0)

    def show(self, double=False):
        """
        Plot the current cross-section of given class
        Args:
            double: Boolean
                when double == True, there are two graphics in a figure

        Returns:
            The cross-section view of the class
        """
        # m is the number of vertices
        m = self.yz.shape[0]
        half_m = int(m/2)
        plt.figure()

        # idx is a 1D numpy array represents the index of points with shape (n,) with value [0, ..., n-1, 0]
        # n is the number of vertices in an individual graphic
        # The figure connect the vertices sequentially, and connect the last point to the first
        if double:
            idx = np.mod(np.arange(half_m + 1), half_m)
            plt.plot(self.yz[idx, 0], self.yz[idx, 1])
            plt.plot(self.yz[idx+half_m, 0], self.yz[idx+half_m, 1], color='#1f77b4')
        else:
            idx = np.mod(np.arange(m + 1), m)
            plt.plot(self.yz[idx, 0], self.yz[idx, 1])

        plt.xlabel('y')
        plt.ylabel('z')
        plt.axis('equal')
        plt.title("Cross-section {:d}, {:d} points".format(self.C.shape[0], self.C.shape[1]))
        plt.show()


class AShapeColumn(CrossSection):

    def setShape(self, w_2, h_1, h_2, h_3, h_4, h_5, k, bridge_thick):
        """
        见示意图

        """
        w_1 = (h_1+h_2+h_3)/k
        yz1 = np.array([
            [0, (h_1+h_2+h_3+h_4+h_5)],
            [0, (h_1+h_2+h_3)],
            [h_3/k, (h_1+h_2)],
            [0, (h_1+h_2)],
            [0, h_1],
            [(h_2+h_3)/k, h_1],
            [w_1, 0],
            [w_1+w_2, 0],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4)],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4+h_5)]
        ])

        yz2 = np.zeros(yz1.shape)
        yz2[:, 0] = -yz1[:, 0]
        yz2[:, 1] = yz1[:, 1]
        yz = np.concatenate((yz1,yz2), 0)
        bridge_max_width = (h_3-bridge_thick)/k * 2
        self.yz = yz
        self.add_current_section()

        return yz, bridge_max_width, h_5


class Rectangle(CrossSection):
    def setShape(self, b, h, h2):
        """
        b - width
        h - height on the top of deck or the height of the columns
        h2 - thickness of the deck
        """
        yz = np.array([
            [-0.5*b,h-h2],
            [0.5*b,h-h2],
            [0.5*b,h],
            [-0.5*b,h]
        ])
        self.yz = yz
        self.add_current_section()
        return yz


class Circle(CrossSection):
    def setShape(self, radius_circle):
        """
        radius_circle - generate a circle
        """
        alpha = np.linspace(0, 2 * np.pi, 50)
        x = np.array(np.cos(alpha)) * radius_circle
        y = np.array(np.sin(alpha)) * radius_circle
        yz = np.zeros((len(x), 2))
        yz[:, 0] = x
        yz[:, 1] = y
        yz = np.delete(yz, -1, axis=0).round(2)
        self.yz = yz
        self.add_current_section()
        return yz


class Member:
    def __init__(self, C, t=None, quat=None):
        """

        Args:
            C: The accumulative cross-section got from an CrossSection instance
            t: Translation of the cross-sections
            quat: Rotation in quaternion of the cross-sections

        Member variables:
            n: Number of cross-sections in the instance
            npts: Vertices number of one cross-section
            t: Translation of the cross-sections
            r: Rotation instance
            v: list of three-tuples (vertices) represented the coordinates in 3D
            f: list of faces
        """
        self.n = C.shape[0]
        self.npts = C.shape[1]
        if t is None:
            t = np.zeros((self.n, 3))
            t[:, 0] = np.arange(self.n)
        self.t = t

        if quat is None:
            quat = np.zeros((self.n, 4))
            quat[:, 3] = 1.  # quaternion is (x,y,z,w)
        self.r = R.from_quat(quat)

        self.v = []
        self.f = []

        xyz = np.zeros((self.npts, 3))
        for i in range(self.n):
            xyz[:, 1:] = C[i, :, :]
            c = self.r[i].apply(xyz) + self.t[np.zeros(self.npts, dtype=int) + i, :]
            self.v = self.v + [(c[k, 0], c[k, 1], c[k, 2]) for k in range(self.npts)]
            if i > 0:
                m = self.npts * (i - 1)
                idx1 = np.arange(m, m + self.npts)
                idx2 = np.arange(m + self.npts, m + 2 * self.npts)
                self.f = self.f + [(idx1[k], idx1[np.mod(k + 1, self.npts)], idx2[np.mod(k + 1, self.npts)], idx2[k])
                                   for k in range(self.npts)]

    def show(self):
        v = np.array(self.v)
        plt.figure()
        ax = plt.axes(projection='3d')
        for f in self.f:
            idxv = np.array(f)
            ax.plot3D(v[idxv, 0], v[idxv, 1], v[idxv, 2], 'k')

        vmin = v.min(0)
        vmax = v.max(0)
        ctr = (vmin + vmax) / 2.
        half = (vmax - vmin).max() / 2.
        ax.set_xlim((ctr[0] - half, ctr[0] + half))
        ax.set_ylim((ctr[1] - half, ctr[1] + half))
        ax.set_zlim((ctr[2] - half, ctr[2] + half))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


if __name__ == "__main__":
    cs = CrossSection()
    n = 5
    # create constant rectangular cross section
    for i in range(n):
        # cs.wt_beam(1.0, 1.0, 0.1, 0.1)
        cs.A_shape_column(1,5,2,15,2,8,4,2)
        # cs.hss_beam(1, 1, 0.2)
        # cs.pipe(3, 1)
    # cs.show()
    cs.show(double=True)

    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n, 3))
    t[:, 0] = np.arange(n)
    a = 1.5
    omega = np.pi / (n - 1)
    #t[:, 2] = np.sin(omega * np.arange(n)) * a
    rotvec = np.zeros((n, 3))
    rotvec[:, 1] = -np.arctan(omega * a * np.cos(omega * np.arange(n)))
    Rot = R.from_rotvec(rotvec)

    #m = Member(cs.C, t, Rot.as_quat())
    m = Member(cs.C, t, None)
    m.show()
