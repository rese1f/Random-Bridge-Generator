import bpy
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import random


class Member:
    """
    define cross section in the local coordinate system
    x is the direction perpendicular to the cross-section
    cross-sections are defined in yz plane
    """

    def __init__(self, cfg, n, t=None, quat=None):
        """load cfg

        Args:
            cfg (str): cfg file path
            n: Number
            t: Translation of the cross-sections
            quat: Rotation in quaternion of the cross-sections

        Member variables:
            n: Number of cross-sections in the instance
            t: Translation of the cross-sections
            r: Rotation instance
            v: list of three-tuples (vertices) represented the coordinates in 3D
            f: list of faces
        """

        self.cfg = cfg
        self.name = self.cfg['name']
        self.shape = self.cfg['shape']

        self.yz = None

        self.f = None
        self.v = None
        self.n = n
        self.t = t
        
        if quat is None:
            quat = np.zeros((self.n, 4))
            quat[:, 3] = 1.  # quaternion is (x,y,z,w)
        self.r = R.from_quat(quat)
    
        self.npts = 0

        self.obj = None

        print("create {}\nshape {}".format(self.name, self.shape))

    def setMember(self):
        self.npts = self.yz.shape[0]
        if self.t is None:
            t = np.zeros((self.n, 3))
            t[:, 0] = np.arange(self.n)
        self.t = t

        if self.quat is None:
            quat = np.zeros((self.n, 4))
            quat[:, 3] = 1.  # quaternion is (x,y,z,w)
        self.r = R.from_quat(quat)

        self.v = []
        self.f = []

        # one layer
        c_one_layer = self.yz.reshape((1, self.yz.shape[0], self.yz.shape[1]))
        C = np.concatenate([c_one_layer for i in range(self.n)], axis=0)
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

    def setMember3d(self):
        return None

    def showCrossSection(self):
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
        half_m = int(m / 2)
        plt.figure()

        # idx is a 1D numpy array represents the index of points with shape (n,) with value [0, ..., n-1, 0]
        # n is the number of vertices in an individual graphic
        # The figure connect the vertices sequentially, and connect the last point to the first
        idx = np.mod(np.arange(m + 1), m)
        plt.plot(self.yz[idx, 0], self.yz[idx, 1])

        plt.xlabel('y')
        plt.ylabel('z')
        plt.axis('equal')
        plt.title("Cross-section of {}, {:d} points".format(self.name, self.yz.shape[0]))
        plt.show()

    def createObj(self, name, obj_num=1):
        vertices = self.v
        edges = []
        faces = self.f
        f1 = ()
        f2 = ()
        for i in range(self.npts):
            f1 += (i,)
            f2 += (self.npts * self.n - i - 1,)
        faces.append(f1)
        faces.append(f2)

        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        obj = bpy.data.objects.new(name, new_mesh)
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(obj)

        self.obj = obj


class Rectangle(Member):
    def __init__(self, cfg, n=1, t=None, quat=None):
        """
        Create a W-beam-shape cross-section
        Args:
            b: Flange length
            h: Web length
        """
        super().__init__(cfg, n, t, quat)
        self.b = self.shape['Flange length']
        self.h = self.shape['Web length']

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


class ConcreteSolid(Member):
    def __init__(self, cfg, n, t=None, quat=None):
        super().__init__(cfg, n, t, quat)
        self.w_deck = self.shape['w_deck']
        self.t_deck = self.shape['t_deck']
        self.h_deck = self.shape['h_deck']

        m = random.uniform(0, 1)

        yz = np.array([
            [(self.w_deck / 2 - m), (self.h_deck)],
            [(self.w_deck / 2), (self.h_deck + self.t_deck / 2)],
            [(self.w_deck / 2 - m), (self.h_deck + self.t_deck)],
            [-(self.w_deck / 2 - m), (self.h_deck + self.t_deck)],
            [-(self.w_deck / 2), (self.h_deck + self.t_deck / 2)],
            [-(self.w_deck / 2 - m), (self.h_deck)]
        ])

        self.yz = yz
        self.setMember()
