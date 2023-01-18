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

# basic Cross-Sectional Part

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
'''
def setCircle2d(radius):
    cfg = {
        "name": "circle",
        "shape": {
            'coordinate of center': coord,
            'radius': radius,
            'element number': num
        }
    }
    return cfg
'''
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

    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(duplicate)
    # # collection_try = bpy.data.collections
    # for co in bpy.data.collections:
    #     collection_try = co
    # bpy.data.collections[collection_try.name].objects.link(duplicate)
    # # bpy.data.collections["Collection"].objects.link(duplicate)

def duplicateDistance(w,l):
    n = int(l/w)
    D = np.zeros(n)
    for i in range(n):
        D[i] = w/2 + w*i
    return D 

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

        self.setMember3d()

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

        self.setMember3d()

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

        self.setMember3d()    

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

        self.setMember3d()


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

        self.setMember3d()

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

        self.setMember3d()      

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


class Circle2d(Member):
    def __init__(self,cfg, n, t, quat,name):
        super().__init__(cfg, n, t, quat)
        self.cfg = cfg
        radius = self.shape['radius']
        alpha = np.linspace(0, 2*np.pi, 50)
        x = np.array(np.cos(alpha))*radius
        y = np.array(np.sin(alpha))*radius
        yz = np.zeros((len(x),2))
        yz[:,0] = x
        yz[:,1] = y
        yz = np.delete(yz,-1,axis=0).round(2)
        self.yz = yz
        self.setMember()



###################################################################################################################
###################################################################################################################
class SuperStructure:
    def __init__(self):
        self.cable = None    
        self.cable_main = None 

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


class Cable2d(SuperStructure):
    def __init__(self, name, length, trans = None, cable_radius = 0.1):
        super().__init__()

        self.cable_radius = cable_radius
        self.length = length
        self.trans = trans #transform

        cfg = setCircle(None, self.cable_radius, None)
        
        member = Circle2d(cfg, self.length, self.trans, None,name)

        member.createObj(name)
        self.cable_main = member.obj
        

class Cable2d_s(SuperStructure):
    def __init__(self, name, length, cable_radius = 0.05):
        super().__init__()  
        self.length = length
        self.cable_radius = cable_radius
        cfg = setCircle(None, self.cable_radius, None)
        
        t = np.zeros((self.length, 3))
        t[:,0] = np.arange(self.length)
        member = Circle2d(cfg, self.length, t, None,name)
        member.createObj(name)

        self.cable = member.obj

# Cable2d_s(10, 'cable', 8, 10, 10, 2, 2)
# Cable2d(10, 'cable', 8, 10, 2, 10, 10, 10)

class SubStructure:
    def __init__(self):
        self.column = None

class Column(SubStructure):
    def __init__(self, w_column, h_column, t_column, l_column, h_deck_t, t_deck, name):
        super().__init__()
        self.W_column = w_column
        self.H_column = h_column
        self.T_column = t_column # latitude
        self.L_column = l_column # longitude
        self.H_deck = h_deck_t
        self.T_deck = t_deck
        self.name = name
        self.cable_function = [0, 0, 0, 0, 0, 0]

    def A1(self):
        cfg_start = setColumnBasic(self.W_column, self.H_column, self.T_column, self.H_deck, self.T_deck) 
        cfg_end = cfg_start

        h_deck = self.H_deck - self.T_deck

        k = self.H_column / (self.W_column/2)
        b_out = self.H_column
        b_in = self.H_column - k * self.T_column
        b_cable = (b_out + b_in) / 2

        w_tria = -2 * (h_deck - b_in) / k
        h_tria = self.H_column - self.T_column * k - h_deck
        H_tria = h_deck + h_tria  

        cfg_hollow_start = setTriangle(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start

        orig =  A1(cfg_start, cfg_end, self.L_column)
        orig.createObj(self.name)
        hollow = Triangle(cfg_hollow_start, cfg_hollow_end, self.L_column + 5)
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0
        self.cable_function[0] = h_deck + (self.H_column - h_deck) / 3
        self.cable_function[1] = self.H_column - self.T_column * k / 2 ## [z_start, z_end, k, b, b_in]         

    def double(self):
        cfg_start = setColumnBasic(self.W_column, self.H_column, self.T_column, self.H_deck, self.T_deck) 
        cfg_end = cfg_start

        h_deck = self.H_deck - self.T_deck
        
        member = Double(cfg_start, cfg_end, self.L_column)
        member.createObj(self.name)

        self.column = member.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.W_column/2 - self.T_column/2
        self.cable_function[0] = h_deck + (self.H_column - h_deck) / 3
        self.cable_function[1] = h_deck + (self.H_column - h_deck) * 9/10

    def door(self):
        cfg_start = setColumnBasic(self.W_column, self.H_column, self.T_column, self.H_deck, self.T_deck) 
        cfg_end = cfg_start
        h_deck = self.H_deck - self.T_deck        

        w_rec = self.W_column - 2 * self.T_column
        h_rec = self.H_column - 3 * self.T_column - h_deck
        H_rec = h_deck + h_rec

        cfg_hollow_start = setRectangle(w_rec, h_rec, H_rec)
        cfg_hollow_end = cfg_hollow_start

        h_deck = self.H_deck - self.T_deck

        orig = Door(cfg_start, cfg_end, self.L_column)
        orig.createObj(self.name)
        hollow = Rectangle(cfg_hollow_start, cfg_hollow_end, self.L_column + 5)        
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.W_column/2 - self.T_column/2
        self.cable_function[0] = h_deck + (self.H_column - h_deck) / 3
        self.cable_function[1] = h_deck + (self.H_column - h_deck) * 9/10 

    def tower(self):
        cfg_start = setColumnBasic(self.W_column, self.H_column, self.T_column, self.H_deck, self.T_deck) 
        cfg_end = cfg_start
        h_deck = self.H_deck - self.T_deck

        a = random.uniform(1/5, 1/3)
        h_cable = a * self.H_column
        b = random.uniform(0.5, 0.8)
        w_bottom = b * self.W_column
        w_top = 2 * self.T_column
        
        k = (self.H_column - h_cable - h_deck) / (self.W_column/2 - self.T_column) 
        b_cable = 0
        b_in = self.H_column - h_cable

        w_tria = self.W_column - 2 * self.T_column
        h_tria = self.H_column - h_cable - h_deck
        H_tria = h_deck + h_tria
        cfg_hollow_start =setTriangle(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start
         
        orig =  Tower(cfg_start, cfg_end, self.L_column)
        orig.createObj(self.name)
        hollow = Triangle(cfg_hollow_start, cfg_hollow_end, self.L_column + 5)
        hollow.createObj('hollow')

        Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0.2
        self.cable_function[0] = self.H_column - h_cable * 9/10
        self.cable_function[1] = self.H_column - h_cable * 1/10 


class Deck:
    def __init__(self):
        self.slab = None

class Slab(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_column,\
         t_column, name, w_deck = None, cable_function = None):
        super().__init__()
        self.T_deck = t_deck
        self.H_deck = h_deck
        self.L_deck = l_deck
        self.W_column = w_column
        self.T_column = t_column
        self.T_truss = None
        self.name = name
        self.cable_function = cable_function

        if cable_function == None:
            self.W_deck = w_deck
        else:
            k = self.cable_function[2]
            b_in = self.cable_function[4]
            h = self.H_deck + self.T_deck
            if b_in == 0:
                self.W_deck = self.W_column - self.T_column * 4/3
            else:
                self.W_deck = - (h - b_in)/k * 2
                
    def concrete_solid(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = ConcreteSolid(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj

    def concrete_PK(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = ConcretePK(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj


    def concrete_box(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = ConcreteBox(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj
   

    def concrete_costalia(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = ConcreteCostalia(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj
    

    def steel_box(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = SteelBox(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj
     

    def steel_sidebox(self):
        cfg_start = setDeckBasic(self.W_deck, self.T_deck, self.H_deck)
        cfg_end = cfg_start

        member = SteelSidebox(cfg_start, cfg_end, self.L_deck)
        member.createObj(self.name)
        self.slab = member.obj

###########

class SuspensionBridgeGenerator:
    def __init__(self, l_deck = 0, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0,\
                 l_column = 2, w_column = 10, h_column = 36):
        #  (self, l_deck = None, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0, l_column = 2, w_column = 10, h_column = 36):    
        
        
        t_column = 1
        index_cable = 1 
        face_cable = 2 
        num_cable = 7 
        truss = 0
        cable_span = 2

        index_column_list = [2,3]
        index_column = random.choice(index_column_list)

        if index_column == 1 or index_column ==2 or index_column == 3:
            index_deck_list = [1,2,4,5,6]
            face_cable = 2
        elif index_column == 4:
            index_deck_list = [3]
            face_cable = 1
        index_deck = random.choice(index_deck_list)    
        
        self.num_column = 2
        self.H_column = h_column
        self.T_column = t_column
        self.L_column = l_column
        self.W_column = w_column
        self.H_deck = h_deck
        self.T_deck = t_deck
        self.index_column = index_column
        self.index_deck = index_deck
        self.index_cable = index_cable
        self.face_cable = face_cable
        self.num_cable = num_cable
        self.cable_span = cable_span
        self.tru = truss
        

        a = random.uniform(3, 6)
        self.dist_column = a * self.H_column
        self.L_deck = 2 * self.dist_column
        
        self.column()
        self.deck()
        #self.cable()
        self.mainCable()

    def fx1(self,x):
        return x-self.dist_column

    def fx2(self,x):
        return x-self.dist_column/2

    def fx3(self,x):
        return x+self.dist_column/2
    
    def fz1(self,x):
        return (0.9*self.H_column-self.H_deck)/(self.dist_column/2)**2*x**2+self.H_deck

    def fz2(self,x):
        return (0.9*self.H_column-1.2*self.H_deck)/(self.dist_column/2)**2*(x-self.dist_column/2)**2+1.2*self.H_deck

    def fz3(self,x):
        return (0.9*self.H_column-self.H_deck)/(self.dist_column/2)**2*(x-self.dist_column/2)**2+self.H_deck


    def column(self):

        for i in range(self.num_column):        
            member = Column(self.W_column, self.H_column, self.T_column, \
                self.L_column, self.H_deck, self.T_deck, 'Column')
            if self.index_column == 1:
                member.A1()
            elif self.index_column == 2:
                member.double()
            elif self.index_column == 3:
                member.door()
            elif self.index_column == 4:
                member.tower()

            self.cable_function = member.cable_function
            member.column.location.x = (-1)**(i+1) * self.dist_column/2


    def deck(self):
        member = Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, \
            self.T_column, 'Deck', None, self.cable_function)
        if self.index_deck == 1:
            member.concrete_solid()
        elif self.index_deck == 2:
            member.concrete_PK()
        elif self.index_deck == 3:
            member.concrete_box()
        elif self.index_deck == 4:
            member.concrete_costalia()
        elif self.index_deck == 5:
            member.steel_box() 
        elif self.index_deck == 6:
            member.steel_sidebox()


    def mainCable(self):

        cables = Cable2d_s("cable_straight", 2, 0.05)

        for i in range(1,4):
            for j in range(1,3):
                if i == 1:
                    functionx = self.fx1
                    function = self.fz1
                    length = int(self.dist_column/2)
                elif i == 2:
                    functionx = self.fx2
                    function = self.fz2
                    length = int(self.dist_column)
                elif i == 3:
                    functionx = self.fx3
                    function = self.fz3
                    length = int(self.dist_column/2) 

                t = np.zeros((length,3))
                t[:,0] = np.array([functionx(i) for i in range(length)])
                t[:,1] = (self.W_column/2 - self.T_column/2) * (-1)**j
                t[:,2] = np.array([function(i) for i in range(length)])
                name = "main_cable" + str(i)
                Cable2d(name, length, t, 0.1)

                cable_loc = duplicateDistance(self.cable_span, length)
                for m,n in enumerate(cable_loc):
                    Duplicate(cables.cable, "cable_straight"+str(m), functionx(n), ((self.W_column/2 - self.T_column/2) * (-1)**j),
                    self.H_deck, 0, -np.pi/2, 0, function(n) - self.H_deck, None, None)





    def cable(self):
        cable_start = np.array([0,0,0])
        cable_end = np.array([1,0,0])
        Cable(cable_start, cable_end, "cable_3", cable_radius = 0.05, num = 12)
        



###### 

h_column = np.random.rand()*10+30

t_column = 1

l_column = 2

w_column = np.random.rand()*5+10

h_deck = h_column/4

t_deck = 1

index_column_list = [2,3]
index_column = random.choice(index_column_list)

if index_column == 1 or index_column ==2 or index_column == 3:
    index_deck_list = [1,2,4,5,6]
    face_cable = 2
elif index_column == 4:
    index_deck_list = [3]
    face_cable = 1
index_deck = random.choice(index_deck_list)


bridge = SuspensionBridgeGenerator(0, h_deck, t_deck, 10, 0, l_column, w_column, h_column)

# l_deck = 0, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0, l_column = 2, w_column = 10, h_column = 36