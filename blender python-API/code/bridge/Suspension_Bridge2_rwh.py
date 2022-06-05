import bpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *

mu = bpy.data.texts["member_util.py"].as_module()

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


class SuspensionBridgeGenerator:
    def fx(self, x):
        return x

    def fy(self, x):
        return 0

    def fz(self, x):
        return 20

    def __init__(self):
        self.Length_of_Bridge = 1001
        self.Width_of_Bridge = 10
        self.Thickness_of_Deck = 2
        self.Thickness_of_Column = 5
        self.column_x1 = 200  # X - coordinate of the first column
        self.column_x2 = 800  # X - coordinate of the second column
        self.Span_of_Columns = self.column_x2 - self.column_x1
        self.Width_of_Column = 4
        self.Height_of_Column = 60
        self.Radius_Main_Cable = 0.2
        self.Span_of_Cable = 10
        self.height = np.array([self.fz(i) for i in range(self.Length_of_Bridge)])

    def fz_cable1(self, x):
        return ((0.9 * self.Height_of_Column - self.fz(x)) / self.column_x1 ** 2) * x ** 2 + self.fz(x)

    def fz_cable2(self, x):
        return (4 * (self.Height_of_Column * 0.9 - self.fz(x) - 6) / self.Span_of_Columns ** 2) * (
                    x - self.Span_of_Columns / 2) ** 2 + self.fz(x) + 6

    def fz_cable3(self, x):
        return ((0.9 * self.Height_of_Column - self.fz(x)) / (self.Length_of_Bridge - self.column_x2) ** 2) * (
                    x - (self.Length_of_Bridge - self.column_x2)) ** 2 + self.fz(x)

    def duplicateDistance(self, w, l):
        n = int(l / w)
        D = np.zeros(n)
        for i in range(n):
            D[i] = w / 2 + w * i
        return D

    def slabBuild(self, name):
        cs = mu.Rectangle()
        n = self.Length_of_Bridge
        thickness_deck = self.Thickness_of_Deck
        height = self.height
        width = self.Width_of_Bridge

        for i in range(n):
            hb = height[i]
            h2 = min(thickness_deck, hb)
            cs.setShape(width, hb, h2)

        m = mu.Member(cs.C, None, None)
        vertices = m.v
        edges = []
        faces = m.f

        f1 = ()
        f2 = ()
        for i in range(m.npts):
            f1 += (i,)
            f2 += (m.npts * n - i - 1,)
        faces.append(f1)
        faces.append(f2)

        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        deckzip = bpy.data.objects.new(name, new_mesh)
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(deckzip)
        return

    def mainColumnBuild(self, name, Right=True, Front=True):
        '''
        Args:
            Width - Width of Column
            Height - Height of Column
            x_dis - x coordinate of Column
            y_dis - y coordinate of Column
        Returns:
            One Column Model
        '''
        cs = mu.Rectangle()
        n = self.Thickness_of_Column
        width = self.Width_of_Column
        height = self.Height_of_Column

        if Right == True:
            x_dis = self.column_x1
        else:
            x_dis = self.column_x2

        if Front == True:
            y_dis = self.Width_of_Bridge / 2
        else:
            y_dis = -self.Width_of_Bridge / 2

        for i in range(n):
            h2 = height
            cs.setShape(width, height, h2)
        m = mu.Member(cs.C, None, None)
        vertices = m.v
        edges = []
        faces = m.f
        f1 = ()
        f2 = ()
        for i in range(m.npts):
            f1 += (i,)
            f2 += (m.npts * n - i - 1,)
        faces.append(f1)
        faces.append(f2)
        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        main_column_1 = bpy.data.objects.new(name, new_mesh)
        main_column_1.location.x = x_dis
        main_column_1.location.y = y_dis
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(main_column_1)

    def mainCableBuild(self, Part, name, Right=True):
        '''
        Args:
            Length - Length of this part of main cable
            radius - radius of main cable (default r = 0.2)
            x_dis - x coordinate of Start of Main Cable
            y_dis - y coordinate of Start of Main Cable
            function - the mathmatical function of main cable in particuler part
        Returns:
            One Column Model
        '''
        cs = mu.Circle()

        if Part == 'Part_1':
            n = self.column_x1
            x_dis = 0
            function = self.fz_cable1
        elif Part == 'Part_2':
            n = self.Span_of_Columns
            x_dis = self.column_x1
            function = self.fz_cable2
        elif Part == 'Part_3':
            n = self.Length_of_Bridge - self.column_x2
            x_dis = self.column_x2
            function = self.fz_cable3
        radius = self.Radius_Main_Cable

        if Right == True:
            y_dis = self.Width_of_Bridge / 2
        else:
            y_dis = -self.Width_of_Bridge / 2

        for i in range(n):
            cs.setShape(radius)
        t = np.zeros((n, 3))
        t[:, 0] = np.array([self.fx(i) for i in range(n)])
        t[:, 1] = np.array([self.fy(i) for i in range(n)])
        t[:, 2] = np.array([function(i) for i in range(n)])
        # rotvec = np.zeros((n,3))
        # rotvec[:,2] = theta
        # Rot = R.from_rotvec(rotvec)

        m = mu.Member(cs.C, t, None)
        vertices = m.v
        edges = []
        faces = m.f

        f1 = ()
        f2 = ()
        for i in range(m.npts):
            f1 += (i,)
            f2 += (m.npts * n - i - 1,)
        faces.append(f1)
        faces.append(f2)
        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        main_cable_1 = bpy.data.objects.new(name, new_mesh)
        main_cable_1.location.x = x_dis
        main_cable_1.location.y = y_dis
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(main_cable_1)
        return

    def cableBuild(self, Part, name, Right=True):
        """
        Args:
            Span - Span distance between two cables
            Length - Length of this part of main cable
            radius - radius of main cable (default r = 0.1)
            x_dis - x coordinate of Start of Main Cable
            y_dis - y coordinate of Start of Main Cable
            function - the mathmatical function of main cable in particuler part
        Returns:
            One Column Model
        """

        cs = mu.Circle()
        Span = self.Span_of_Cable
        radius = self.Radius_Main_Cable / 2
        if Part == 'Part_1':
            Length = self.column_x1
            x_dis = 0
            function = self.fz_cable1
        elif Part == 'Part_2':
            Length = self.Span_of_Columns
            x_dis = self.column_x1
            function = self.fz_cable2
        elif Part == 'Part_3':
            Length = self.Length_of_Bridge - self.column_x2
            x_dis = self.column_x2
            function = self.fz_cable3

        if Right == True:
            y_dis = self.Width_of_Bridge / 2
        else:
            y_dis = -self.Width_of_Bridge / 2

        n = 2  # Set the length of cable as 1 first, then use [scale] to change the length
        for i in range(n):
            cs.setShape(radius)
        m = mu.Member(cs.C, None, None)
        vertices = m.v
        edges = []
        faces = m.f

        f1 = ()
        f2 = ()
        for i in range(m.npts):
            f1 += (i,)
            f2 += (m.npts * n - i - 1,)
        faces.append(f1)
        faces.append(f2)

        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        cable = bpy.data.objects.new(name, new_mesh)
        cable.rotation_euler[1] = -np.pi / 2
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(cable)

        for j, i in enumerate(self.duplicateDistance(Span, Length)):
            if j == 0:
                cable.scale[0] = 0
                cable.location.x = x_dis + i
                cable.location.y = y_dis
                cable.location.z = self.fz(i)
            else:
                cable = bpy.data.objects[name]
                namezip = str(name) + str(j)
                cablezip = bpy.data.objects.new(namezip, cable.data)
                cablezip.location.x = x_dis + i
                cablezip.location.y = y_dis
                cablezip.location.z = self.fz(i)
                cablezip.rotation_euler[1] = - np.pi / 2
                cablezip.scale[0] = function(i) - self.fz(i)
                bpy.data.collections["Collection"].objects.link(cablezip)
        return

    def Assemble(self):
        self.slabBuild("Slab")
        self.mainColumnBuild('Column1', Right=True, Front=True)
        self.mainColumnBuild('Column2', Right=True, Front=False)
        self.mainColumnBuild('Column3', Right=False, Front=True)
        self.mainColumnBuild('Column4', Right=False, Front=False)
        self.mainCableBuild('Part_1', 'MainCable1', Right=True)
        self.mainCableBuild('Part_2', 'MainCable2', Right=True)
        self.mainCableBuild('Part_3', 'MainCable3', Right=True)
        self.mainCableBuild('Part_1', 'MainCable4', Right=False)
        self.mainCableBuild('Part_2', 'MainCable5', Right=False)
        self.mainCableBuild('Part_3', 'MainCable6', Right=False)
        self.cableBuild('Part_1', 'Cable0', Right=True)
        self.cableBuild('Part_2', 'Cable2', Right=True)
        self.cableBuild('Part_3', 'Cable3', Right=True)
        self.cableBuild('Part_1', 'Cable4', Right=False)
        self.cableBuild('Part_2', 'Cable5', Right=False)
        self.cableBuild('Part_3', 'Cable6', Right=False)


bridge = SuspensionBridgeGenerator()
bridge.Assemble()
