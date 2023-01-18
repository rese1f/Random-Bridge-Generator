import tracemalloc
from types import NoneType
import bpy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

###################################################################################################################
###################################################################################################################

def setDeckBasic(w_deck, t_deck, h_deck):
    cfg = {
        "name": "deck_basic",
        "shape": {
            'deck width': w_deck,
            'deck thickness': t_deck,
            'deck top surface height': h_deck
        }
    }
    return cfg


def setColumnBasic(w_column, h_column, t_column, h_deck, t_deck):
    cfg = {
        "name": "column_basic",
        "shape": {
            'column width': w_column,
            'column height': h_column,
            'column thickness': t_column,
            'deck top surface height': h_deck,
            'deck thickness': t_deck
        }
    }
    return cfg


def setTriangle(w, h, H):
    cfg = {
        "name": "triangle",
        "shape": {
            'bottom width': w,
            'height': h,
            'height of top': H
        }
    }
    return cfg


def setTriangle2(a1, b1, a2, b2, a3, b3, c):
    cfg = {
        "name": "triangle2",
        "shape": {
            'x1': a1,
            'x2': a2,
            'x3': a3,
            'z1': b1,
            'z2': b2,
            'z3': b3,
            'y': c
        }
    }
    return cfg


def setRectangle(w, h, H):
    cfg = {
        "name": "rectangle",
        "shape": {
            'bottom width': w,
            'height': h,
            'height of top': H
        }
    }
    return cfg


def setCircle(coord, radius = 0.05, num = 5):
    cfg = {
        "name": "circle",
        "shape": {
            'coordinate of center': coord,
            'radius': radius,
            'element number': num
        }
    }
    return cfg



###################################################################################################################
###################################################################################################################

class Member:
    """
    define cross section in the local coordinate system
    x is the direction perpendicular to the cross-section
    cross-sections are defined in yz plane
    """

    def __init__(self, cfg, n=1, t=None, quat=None):
        """load cfg

        Args:
            cfg (str): cfg file path
            n: Number of faces (thickness)
            t: Translation of the cross-sections
            quat: Rotation in quaternion of the cross-sections

        Member variables:
            n: Number of cross-sections in the instance
            t: Translation of the cross-sections
            r: Rotation instance
            v: list of three-tuples (vertices) represented the coordinates in 3D
            f: list of faces
            cable_function: function of cable in yz direction: [z_start, z_end, k, b, b_in, y_cable] (y_cable is activate if there's no function for cable)
        """

        self.cfg = cfg

        self.name = self.cfg['name']
        self.shape = self.cfg['shape']

        self.yz_end = None
        self.yz = None
        self.type = None

        self.f = None
        self.v = None
        self.n = n
        self.t = t
        self.quat = quat
        self.r = None
        self.npts = 0

        self.obj = None

        # print("create {}\nshape {}".format(self.name, self.shape))

    def setMember(self):
        """
        args - yz
        return - self.v
                 self.f
        """
        self.npts = self.yz.shape[0]
        if self.t is None:
            t = np.zeros((self.n, 3))
            t[:, 0] = np.arange(self.n)
            self.t = t

        if self.quat is None:
            self.quat = np.zeros((self.n, 4))
            self.quat[:, 3] = 1.  # quaternion is (x,y,z,w)
        
        print(self.quat)
        self.r = R.from_quat(self.quat)

        self.v = []
        self.f = []

        # one layer
        c_one_layer = self.yz.reshape((1, self.yz.shape[0], self.yz.shape[1]))
        C = np.concatenate([c_one_layer for i in range(self.n)], axis=0)
        xyz = np.zeros((self.npts, 3))
        
        # print('yz = ' + str(self.yz))
        # print('a = ' + str(xyz[:, 1:].shape))
        # print('b = ' + str(C[1, :, :].shape))
        # print('m = ' + str(c_one_layer))
        # print(self.r)
        # print(self.t)

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
        f1 = ()
        f2 = ()
        for i in range(self.npts):
            f1 += (i,)
            f2 += (self.npts * self.n - i - 1,)
        self.f.append(f1)
        self.f.append(f2)


    def setMember3d(self):
        """
        args - yz
        return - self.v
                 self.f
        """ 
        self.v = []
        self.f = []

        if self.three_d == False:
            start = np.zeros([self.yz.shape[0], 3])
            start[:, 0] = -self.n/2
            start[:, 1:] = self.yz[:, :]
            
            end = np.zeros([self.yz_end.shape[0], 3])
            end[:, 0] = self.n/2
            end[:, 1:] = self.yz_end[:, :]
        
        else:
            start = self.yz
            
            end = self.yz_end        

        for i1 in range(start.shape[0]):
            self.v.append(start[i1, :])

        for i2 in range(end.shape[0]):
            self.v.append(end[i2, :])

        npts = start.shape[0]

        for j in range(npts):
            self.f = self.f + [(j, np.mod(j+1,npts),np.mod(j+1,npts) + npts, j + npts) for k in range(npts)]

        f1 = ()
        f2 = ()
        for i in range(npts):
            f1 += (i,)
            f2 += (npts + i,)
        self.f.append(f1)
        self.f.append(f2)



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
        plt.title("Cross-section of {}, {:d} points".format(self.type, self.yz.shape[0]))
        plt.show()

    def createObj(self, name, obj_num=1):
        vertices = self.v
        edges = []
        faces = self.f
 

        new_mesh = bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices, edges, faces)
        new_mesh.update()
        obj = bpy.data.objects.new(name, new_mesh)
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(obj)

        self.obj = obj

###################################################################################################################
###################################################################################################################

def Hollow(big_obj, small_obj):
    hollow = big_obj.modifiers.new("MyModifier", "BOOLEAN")
    hollow.object = small_obj

    # blender need first activate the object, then apply modifier
    bpy.context.view_layer.objects.active = big_obj
    bpy.ops.object.modifier_apply(modifier="MyModifier")

    # blender need first choose the object, then delete
    small_obj.select_set(True)
    bpy.ops.object.delete()

def Merge(object):
    # bpy.ops.object.select_all(action='DESELECT')#取消选择
    k = len(object)
    bpy.context.view_layer.objects.active=object[0]
    for i in range(k-1):
        bpy.ops.object.select_all(action='DESELECT')#取消选择
        object[0].select_set(True)
        object[i+1].select_set(True)
        bpy.ops.object.join()

def Duplicate(obj, dupli_name, x=None, y=None, z=None, rot_x=None, rot_y=None, rot_z=None, scale_x=None, scale_y=None, scale_z=None):
    duplicate = bpy.data.objects.new(dupli_name, obj.data)

    if x!=None:
        duplicate.location.x = x
    if y!=None:
        duplicate.location.y = y    
    if z!=None:
        duplicate.location.z = z

    if rot_x!=None:
        duplicate.rotation_euler[0] = rot_x
    if rot_y!=None:
        duplicate.rotation_euler[1] = rot_y    
    if rot_z!=None:
        duplicate.rotation_euler[2] = rot_z    
    
    if scale_x!=None:
        duplicate.scale[0] = scale_x    
    if scale_y!=None:
        duplicate.scale[1] = scale_y
    if scale_z!=None:
        duplicate.scale[2] = scale_z

    bpy.data.collections["Collection"].objects.link(duplicate)

###################################################################################################################
###################################################################################################################

class ConcreteSolid(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(0, 1)
        self.yz = np.array([
            [(w_deck/2 - m), (h_deck)],
            [(w_deck/2), (h_deck + t_deck/2)],
            [(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck + t_deck/2)],
            [-(w_deck/2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

class ConcretePK(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck/8, w_deck/4)
        self.yz = np.array([
            [(w_deck/2 - m), (h_deck)],
            [(w_deck/2), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck + t_deck)],
            [-(w_deck/2 - m), (h_deck)]
        ])          

        self.yz_end =  self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

class ConcreteBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)
        
        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']
        
        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck/8, w_deck/4)
        m2 = m/2
        self.yz = np.array([
            [(w_deck/2 - m), (h_deck)],
            [(w_deck/2 - m2), (h_deck + t_deck - 0.2)],
            [(w_deck/2), (h_deck + t_deck - 0.2)],
            [(w_deck/2), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck  + t_deck)],
            [-(w_deck/2), (h_deck + t_deck - 0.2)],
            [-(w_deck/2 - m2), (h_deck + t_deck - 0.2)],
            [-(w_deck/2 - m), (h_deck)]
        ])

        self.yz_end = self.yz


        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()       


class ConcreteCostalia(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(0, 1)

        self.yz = np.array([
            [(w_deck/2), (h_deck)],
            [(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2 - m), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck)]
        ]) 

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()


class SteelBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck/6, w_deck/4)
        m2 = random.uniform(w_deck/18, w_deck/12)        

        self.yz = np.array([
            [(w_deck/2 - m), (h_deck)],
            [(w_deck/2), (h_deck + t_deck/2)],
            [(w_deck/2 - m2), (h_deck + t_deck)],
            [-(w_deck/2 - m2), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck + t_deck/2)],
            [-(w_deck/2 - m), (h_deck)]
        ])              
        
        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

class SteelSidebox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        h_deck = h_deck_t - t_deck # height at bottom of deck
        
        self.yz = np.array([
            [(w_deck/2), (h_deck)],
            [(w_deck/2), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck + t_deck)],
            [-(w_deck/2), (h_deck)]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()          


class Triangle(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']   

        w = self.shape['bottom width']
        h = self.shape['height'] 
        H = self.shape['height of top']

        H = H-h # height of bottom

        self.yz = np.array([
            [(w/2), (H)],
            [0, (H+h)],
            [-(w/2), (H)]
        ])    

        self.yz_end = self.yz

        self.setMember3d()


class Triangle2(Member):
    def __init__(self, cfg, cfg_end, n, three_d=True, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.cfg_end = cfg_end
        self.three_d = three_d
        self.shape_end = self.cfg_end['shape']          

        a1 = self.shape['x1']
        a2 = self.shape['x2']
        a3 = self.shape['x3']
        b1 = self.shape['z1']
        b2 = self.shape['z2']
        b3 = self.shape['z3']
        c_start = self.shape['y']

        self.yz = np.array([
        [a1, c_start, b1],
        [a2, c_start, b2],
        [a3, c_start, b3]
        ])

        a1 = self.shape_end['x1']
        a2 = self.shape_end['x2']
        a3 = self.shape_end['x3']
        b1 = self.shape_end['z1']
        b2 = self.shape_end['z2']
        b3 = self.shape_end['z3']
        c_end = self.shape_end['y']

        self.yz_end = np.array([
        [a1, c_end, b1],
        [a2, c_end, b2],
        [a3, c_end, b3]
        ])        

        self.setMember3d()

class Rectangle(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w = self.shape['bottom width']
        h = self.shape['height']
        H = self.shape['height of top']

        H = H - h   

        self.yz = np.array([
            [w/2, H],
            [w/2, H+h],
            [-w/2, H+h],
            [-w/2, H]
        ])  

        self.yz_end = self.yz

        self.setMember3d()


class A1(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat) 

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w_column = self.shape['column width']
        h_column = self.shape['column height']
        t_column = self.shape['column thickness']
        h_deck_t = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        h_deck = h_deck_t - t_deck

        k = h_column / (w_column/2)
        b_out = h_column
        b_in = h_column - k * t_column
        b_cable = (b_out + b_in) / 2
        
        self.yz = np.array([
            [-((h_deck - t_column - b_in)/k), (h_deck - t_column)],
            [(w_column/2 - t_column), 0],
            [(w_column/2), 0],
            [0, h_column],
            [-(w_column/2), 0],
            [-(w_column/2 - t_column), 0],
            [((h_deck - t_column - b_in)/k), (h_deck - t_column)]
        ])

        self.yz_end = self.yz

        self.setMember3d()

class Double(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']         

        w_column = self.shape['column width']
        h_column = self.shape['column height']
        t_column = self.shape['column thickness']
        h_deck_t = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        h_deck = h_deck_t - t_deck

        self.yz = np.array([
            [(w_column/2 - t_column), (h_deck - t_column)],
            [(w_column/2 - t_column), 0],
            [w_column/2, 0],
            [w_column/2, h_column],
            [(w_column/2 - t_column), h_column],
            [(w_column/2 - t_column), h_deck],
            [-(w_column/2 - t_column), h_deck],
            [-(w_column/2 - t_column), h_column],
            [-w_column/2, h_column],
            [-w_column/2, 0],
            [-(w_column/2 - t_column), 0],
            [-(w_column/2 - t_column), (h_deck - t_column)]
        ])        

        self.yz_end = self.yz

        self.setMember3d()


class Door(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']         

        w_column = self.shape['column width']
        h_column = self.shape['column height']
        t_column = self.shape['column thickness']
        h_deck_t = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        h_deck = h_deck_t - t_deck

        self.yz = np.array([
            [(w_column/2 - t_column), (h_deck - t_column)],
            [(w_column/2 - t_column), 0],
            [w_column/2, 0],
            [w_column/2, h_column],
            [-w_column/2, h_column],
            [-w_column/2, 0],
            [-(w_column/2 - t_column), 0],
            [-(w_column/2 - t_column), (h_deck - t_column)]
        ])    

        self.yz_end = self.yz      

        self.setMember3d()

class Tower(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape'] 

        w_column = self.shape['column width']
        h_column = self.shape['column height']
        t_column = self.shape['column thickness']
        h_deck_t = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        h_deck = h_deck_t - t_deck 

        a = random.uniform(1/5, 1/3)
        h_cable = a * h_column
        b = random.uniform(0.5, 0.8)
        w_bottom = b * w_column
        w_top = 2 * t_column
        
        k = (h_column - h_cable - h_deck) / (w_column/2 - t_column) 
        b_cable = 0
        b_in = h_column - h_cable        

        self.yz = np.array([
            [w_bottom/2, 0],
            [w_column/2, h_deck],
            [w_top/2, h_column - h_cable],
            [w_top/2, h_column],
            [-w_top/2, h_column],
            [-w_top/2, h_column - h_cable],
            [-w_column/2, h_deck],
            [-w_bottom/2, 0],
        ])    
        self.yz_end = self.yz                     

        self.setMember3d()

class Circle(Member):
    def __init__(self, cfg, cfg_end, n, three_d = True, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.cfg_end = cfg_end
        self.three_d = three_d
        self.shape_end = self.cfg_end['shape']     

        coord = self.shape['coordinate of center']
        radius = self.shape['radius']
        num = self.shape['element number']

        alpha = np.linspace(0, 2*np.pi, num)
        x = float(coord[0]) 
        y = float(coord[1]) + np.array(np.cos(alpha))*radius
        z = float(coord[2]) + np.array(np.sin(alpha))*radius
        xyz = np.zeros((len(y), 3))
        xyz[:, 0] = x + y*0
        xyz[:, 1] = y
        xyz[:, 2] = z

        self.yz = xyz

        coord = self.shape_end['coordinate of center']
        radius = self.shape_end['radius']
        num = self.shape_end['element number']

        alpha = np.linspace(0, 2*np.pi, num)
        x = float(coord[0]) 
        y = float(coord[1]) + np.array(np.cos(alpha))*radius
        z = float(coord[2]) + np.array(np.sin(alpha))*radius
        xyz = np.zeros((len(y), 3))
        xyz[:, 0] = x + y*0
        xyz[:, 1] = y
        xyz[:, 2] = z

        self.yz_end = xyz   

        self.setMember3d()

class PiColumn(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t = None, quat = None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape'] 

        w_column = self.shape['column width']
        h_column = self.shape['column height']
        t_column = self.shape['column thickness']
        h_deck_t = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        h_deck = h_deck_t - t_deck

        self.yz = np.array([
            [w_column/6, h_column-t_column],
            [w_column/6, 0],
            [w_column/6+1.5*t_column, 0],
            [w_column/6+1.5*t_column, h_column-t_column],
            [w_column/2, h_column-t_column],
            [w_column/2, h_column],
            [-w_column/2, h_column],
            [-w_column/2, h_column-t_column],
            [-(w_column/6+1.5*t_column), h_column-t_column],
            [-(w_column/6+1.5*t_column), 0],
            [-w_column/6, 0],
            [-w_column/6, h_column-t_column]
        ])
        
        self.yz_end = self.yz                     

        self.setMember3d()



###################################################################################################################
###################################################################################################################

class SuperStructure:
    def __init__(self):
        self.cable = None

class Cable(SuperStructure):
    def __init__(self, cable_start, cable_end, name, cable_radius = 0.05, num = 8):
        super().__init__()

        self.cable_start = cable_start
        self.cable_end = cable_end
        self.cable_radius = cable_radius
        self.cable_fit_num = num
        self.name = name

        cfg_start = setCircle(self.cable_start, self.cable_radius, self.cable_fit_num)
        cfg_end = setCircle(self.cable_end, self.cable_radius, self.cable_fit_num)

        member = Circle(cfg_start, cfg_end, 1, True, None, None)
        member.createObj(name)
        self.cable = member.obj






class SubStructure:
    def __init__(self):
        self.column = None

class Column(SubStructure):
    def __init__(self, w_column, h_column, t_column, l_column, h_deck_t, t_deck, name):
        super().__init__()
        self.column_width = w_column
        self.column_height = h_column
        self.column_thickness = t_column # latitude
        self.column_length = l_column # longitude
        self.deck_height = h_deck_t
        self.deck_thickness = t_deck
        self.name = name
        self.cable_function = [0, 0, 0, 0, 0, 0]

    def A1(self):
        cfg_start = setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start

        h_deck = self.deck_height - self.deck_thickness

        k = self.column_height / (self.column_width/2)
        b_out = self.column_height
        b_in = self.column_height - k * self.column_thickness
        b_cable = (b_out + b_in) / 2

        w_tria = -2 * (h_deck - b_in) / k
        h_tria = self.column_height - self.column_thickness * k - h_deck
        H_tria = h_deck + h_tria  

        cfg_hollow_start = setTriangle(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start

        orig =  A1(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = Triangle(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = self.column_height - self.column_thickness * k / 2 ## [z_start, z_end, k, b, b_in]         


    def double(self):
        cfg_start = setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start

        h_deck = self.deck_height - self.deck_thickness
        
        member = Double(cfg_start, cfg_end, self.column_length)
        member.createObj(self.name)

        self.column = member.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.column_width/2 - self.column_thickness/2
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = h_deck + (self.column_height - h_deck) * 9/10


    def door(self):
        cfg_start = setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness        

        w_rec = self.column_width - 2 * self.column_thickness
        h_rec = self.column_height - 3 * self.column_thickness - h_deck
        H_rec = h_deck + h_rec

        cfg_hollow_start = setRectangle(w_rec, h_rec, H_rec)
        cfg_hollow_end = cfg_hollow_start

        h_deck = self.deck_height - self.deck_thickness

        orig = Door(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = Rectangle(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)        
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.column_width/2 - self.column_thickness/2
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = h_deck + (self.column_height - h_deck) * 9/10 
         


    def tower(self):
        cfg_start = setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness

        a = random.uniform(1/5, 1/3)
        h_cable = a * self.column_height
        b = random.uniform(0.5, 0.8)
        w_bottom = b * self.column_width
        w_top = 2 * self.column_thickness
        
        k = (self.column_height - h_cable - h_deck) / (self.column_width/2 - self.column_thickness) 
        b_cable = 0
        b_in = self.column_height - h_cable

        w_tria = self.column_width - 2 * self.column_thickness
        h_tria = self.column_height - h_cable - h_deck
        H_tria = h_deck + h_tria
        cfg_hollow_start =setTriangle(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start
         
        orig =  Tower(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = Triangle(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0.2
        self.cable_function[0] = self.column_height - h_cable * 9/10
        self.cable_function[1] = self.column_height - h_cable * 1/10 

    def piColumn(self):
        cfg_start = setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness
        column = PiColumn(cfg_start, cfg_end, self.column_length)
        column.createObj(self.name)
        self.column = column.obj



# a = Column(10, 10, 1, 3, 11, 1, 'column')
# a.piColumn()
# a.column.location.x = 5
# a.column.location.y = 15   



class Deck:
    def __init__(self):
        self.slab = None
    
class Slab(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_column, t_column, name, tran = None, quat = None, w_deck = None, cable_function = None):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        self.deck_length = l_deck
        self.column_width = w_column
        self.column_thickness = t_column
        self.truss_thickness = None

        self.tran = tran
        self.quat = quat

        self.name = name
        self.cable_function = cable_function

        if cable_function == None:
            self.deck_width = w_deck
        else:
            k = self.cable_function[2]
            b_in = self.cable_function[4]
            h = self.deck_height + self.deck_thickness
            if b_in == 0:
                self.deck_width = self.column_width - self.column_thickness * 2
            else:
                self.deck_width = - (h - b_in)/k * 2
    
    def concrete_solid(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteSolid(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
 

    def concrete_PK(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcretePK(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj


    def concrete_box(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
   

    def concrete_costalia(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteCostalia(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
    

    def steel_box(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = SteelBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
     

    def steel_sidebox(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = SteelSidebox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
    
    def truss(self):
        T_truss = 5
        self.truss_thickness = T_truss
        h = self.deck_height - self.deck_thickness + T_truss
        if self.cable_function != None:
            k = self.cable_function[2]
            b_in = self.cable_function[4]
            if b_in == 0:
                self.deck_width = self.column_width - self.column_thickness * 2
            else:
                self.deck_width = - (h - b_in)/k * 2
        
        v_width = 0.25
        h_width = 0.25

        cfg_start = setRectangle(self.deck_width, self.truss_thickness, h)  
        cfg_end = cfg_start

        cfg_hollow_start = setRectangle(self.deck_width - 2*h_width, self.truss_thickness - 2*v_width, h - v_width) 
        cfg_hollow_end = cfg_hollow_start

        orig = Rectangle(cfg_start, cfg_end, self.deck_length)
        orig.createObj(self.name)
        hollow = Rectangle(cfg_hollow_start, cfg_hollow_end, self.deck_length + 5)
        hollow.createObj('hollow') 

        Hollow(orig.obj, hollow.obj)

        thick_bar = v_width * 1.5
        l = self.deck_length - thick_bar
        width_bar = 4
        height_bar = T_truss

        a11 = -l/2 + thick_bar/2
        b11 = thick_bar + self.deck_height - self.deck_thickness
        a12 = -l/2 + width_bar - thick_bar/2
        b12 = thick_bar+ self.deck_height - self.deck_thickness
        a13 = -l/2 + thick_bar/2
        b13 = height_bar - thick_bar -  height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness

        a23 = -l/2 + thick_bar/2
        b23 = height_bar - thick_bar + self.deck_height - self.deck_thickness
        a22 = -l/2 + width_bar - thick_bar/2
        b22 = height_bar - thick_bar + self.deck_height - self.deck_thickness
        a21 = -l/2 + width_bar - thick_bar/2 
        b21 = thick_bar + height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness    

        for i in range(int(l/width_bar)):
        # for i in range(1):    
            name = 'down_hollow' + str(i)
            cfg1_start = setTriangle2(a11 + width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, 10) 
            cfg1_end = setTriangle2(a11+ width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, -10)
            tria1 = Triangle2(cfg1_start, cfg1_end, 1)
            tria1.createObj(name)
            Hollow(orig.obj, tria1.obj)

        for i in range(int(l/width_bar)):
        # for i in range(1):
            name = 'up_hollow' + str(i)
            cfg2_start = setTriangle2(a21 + width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, 10) 
            cfg2_end = setTriangle2(a21+ width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, -10)
            tria2 = Triangle2(cfg2_start, cfg2_end, 1)
            tria2.createObj(name)
            Hollow(orig.obj, tria2.obj)             

        self.slab = orig.obj


# b = Slab(1, 8, 50, 10, 1, 'slab', None, None, 5)
# b.concrete_costalia()
# b.slab.rotation_euler[2] = -math.pi/3
# b.slab.location.x = 0


class Bearing:
    def __init__(self):
        self.cable_base = None
        self.cable_top = None
        self.column_bearing = None

class CableBase(Bearing):
    def __init__(self, t1, t2, t3, r2, name, turn = 1):                         
        cfg_start = setCircle([(0 - t3*2/3) * turn, 0, 0], r2, 50)
        cfg_end = setCircle([(t3*1/3 + t2) * turn, 0, 0], r2, 50)
        member = Circle(cfg_start, cfg_end, 1)
        member.createObj(name)
        self.cable_base = member.obj

class CableTop(Bearing):
    def __init__(self, t1, t2, t3, r1, name, turn = 1):                         
        cfg_start = setCircle([-(t3*3/4 + t2) * turn, 0, 0], r1, 50)
        cfg_end = setCircle([-(0 - t3/4) * turn, 0, 0], r1, 50)
        member = Circle(cfg_start, cfg_end, 1)
        member.createObj(name)
        self.cable_base = member.obj   

class ColumnBearing(Bearing):
    def __init__(self, a, d, T, t, name):
        cfg_start = setRectangle(a, (T-t)/2, (T-t)/2+t/2)
        cfg_end = cfg_start
        rec1 = Rectangle(cfg_start, cfg_end, a) 
        rec1.createObj('rec1')
        
        cfg_start = setRectangle(a, (T-t)/2, -t/2)
        cfg_end = cfg_start
        rec2 = Rectangle(cfg_start, cfg_end, a) 
        rec2.createObj('rec2')
        
        cfg_start = setCircle([-t/2, 0, 0], d/2, 50)
        cfg_end = setCircle([t/2, 0, 0], d/2, 50)
        column_bearing = Circle(cfg_start, cfg_end, 1)
        column_bearing.createObj(name)
        column_bearing.obj.rotation_euler[1] = math.pi/2
        
        Merge([column_bearing.obj, rec1.obj, rec2.obj])
        self.column_bearing = column_bearing.obj


#################################################################################################################
#################################################################################################################



class SlabBridge:
    def __init__(self, l_deck = 500, h_deck = 11, t_deck = 1, w_deck = 10, d_column = 10, l_column = 2, w_column = 6, h_column = 10):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_height = h_column
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column

        self.bearing_thickness = 0.3

        self.column_number = int(self.deck_length/self.column_distance)-1 # 49
        self.column_thickness = 1

        index_deck_list = [1,2,3,4,5,6]
        self.deck_index = random.choice(index_deck_list)

        self.a = (np.random.rand())*20
        self.b = (np.random.rand())*1000

        self.theta = np.array([np.arctan(float(self.fyd(i,self.a,self.b))) for i in range(1,self.deck_length+1)])


        self.column()
        self.deck()
        self.bearing()

    def fx(self,x,a,b):
        return x

    def fy(self,x,a,b):
        return np.sin(x/b)*a

    def fyd(self,x,a,b):
        fyd = (self.fy(x+1e-9,self.a,self.b)-self.fy(x,self.a,self.b))/1e-9
        return fyd

    def fz(self,a,b):
        x = np.arange(500+1)
        a = abs(np.random.rand())*5+9
        b = abs(np.random.rand())*5+9
        k = (b-a)/100
        height = x*k+a
        for i in range(len(height)):
            if height[i] <= 5:
                height[i] = 5
            elif height[i] >= 14:
                height[i] = 14
        return height 

    def column(self):
        Span = self.column_distance
        Length = self.deck_length
        name = 'column'

        self.column_height -= self.bearing_thickness
        member = Column(self.column_width, self.column_height, self.column_thickness, self.column_length,  self.deck_height, self.deck_thickness, 'pi column')
        member.piColumn()        
        for i in range(self.column_number): 

            Duplicate(member.column, 'Pi_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(float(self.fyd(int(i*self.column_distance),self.a,self.b))))            

        member.column.select_set(True)
        bpy.ops.object.delete()
        self.column_height += self.bearing_thickness


    def deck(self):
        n = self.deck_length
        t = np.zeros((n,3))
        t[:,0] = np.array([self.fx(i,self.a,self.b) for i in range(n)])
        t[:,1] = np.array([self.fy(i,self.a,self.b) for i in range(n)])
        rotvec = np.zeros((n,3))
        rotvec[:,2] = self.theta
        Rot = R.from_rotvec(rotvec)
        quat = Rot.as_quat()
        
        if self.deck_index == 1:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete solid deck', t, quat, self.deck_width, None)
            member.concrete_solid()
        
        elif self.deck_index == 2:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete PK deck', t, quat, self.deck_width, None)
            member.concrete_PK()
        
        elif self.deck_index == 3:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete box deck', t, quat, self.deck_width, None)
            member.concrete_box()
        
        elif self.deck_index == 4:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete constalia deck', t, quat, self.deck_width, None)
            member.concrete_costalia()

        elif self.deck_index == 5:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel box deck', t, quat, self.deck_width, None)
            member.steel_box() 

        elif self.deck_index == 6:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel sidebox deck', t, quat, self.deck_width, None)
            member.steel_sidebox()       
    
    def bearing(self):
        a = self.column_length
        T = self.bearing_thickness
        t = T/3
        d = a

        member = ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
            dis = a/2 + 0.5

            theta = float(self.fyd(i*self.column_distance,self.a,self.b))
            # member1 = ColumnBearing(a, d, T, t, "bearing" + str(i*2+1))
            # member1.column_bearing.rotation_euler[2] = np.arctan(theta)
            # member1.column_bearing.location.x = self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta)
            # member1.column_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            # member1.column_bearing.location.y = self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta)

            # member2 = ColumnBearing(a, d, T, t, "bearing" + str(i*2+2))
            # member2.column_bearing.rotation_euler[2] = np.arctan(theta) 
            # member2.column_bearing.location.x = self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta)
            # member2.column_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            # member2.column_bearing.location.y = self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta)  

        
            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta), self.deck_height - self.bearing_thickness/2 - self.deck_thickness,\
                     0, math.pi/2, np.arctan(theta))

            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta), self.deck_height - self.bearing_thickness/2 - self.deck_thickness,\
                     0, math.pi/2, np.arctan(theta))
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()
        
SlabBridge()          
