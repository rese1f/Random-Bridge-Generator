# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:13:12 2022

@author: dell
"""

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

def setBeamBasic(w_beam, h_beam, t_webplate, t_flange, h_deck, t_deck):
    cfg = {
        "name": "beam_basic",
        "shape": {
            'beam width': w_beam,
            'beam height': h_beam,
            'webplate thickness': t_webplate,
            'flange thickness': t_flange,
            'deck top surface height': h_deck,
            'deck thickness': t_deck       
        }
    }
    return cfg    

def setTriangle(a, b1, c1, b2, c2, b3, c3):
    cfg = {
        "name": "triangle2",
        "shape": {
            'y1': b1,
            'y2': b2,
            'y3': b3,
            'z1': c1,
            'z2': c2,
            'z3': c3,
            'x': a
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


class IBeam(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w_beam = self.shape['beam width']
        h_beam = self.shape['beam height']
        t_webplate = self.shape['webplate thickness']
        t_flange = self.shape['flange thickness']
        h_deck = self.shape['deck top surface height']
        t_deck = self.shape['deck thickness']

        self.yz = np.array([
            [(w_beam/2), (h_deck-t_deck-h_beam)],
            [(w_beam/2), (h_deck-t_deck-h_beam+t_flange)],
            [(t_webplate/2), (h_deck-t_deck-h_beam+t_flange)],
            [(t_webplate/2), (h_deck-t_deck-t_flange)],
            [(w_beam/2), (h_deck-t_deck-t_flange)],
            [(w_beam/2), (h_deck-t_deck)],
            [-(w_beam/2), (h_deck-t_deck)],
            [-(w_beam/2), (h_deck-t_deck-t_flange)],
            [-(t_webplate/2), (h_deck-t_deck-t_flange)],
            [-(t_webplate/2), (h_deck-t_deck-h_beam+t_flange)],
            [-(w_beam/2), (h_deck-t_deck-h_beam+t_flange)],
            [-(w_beam/2), (h_deck-t_deck-h_beam)]
        ])              
        
        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()



class Triangle(Member):
    def __init__(self, cfg, cfg_end, n, three_d=True, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.cfg_end = cfg_end
        self.three_d = three_d
        self.shape_end = self.cfg_end['shape']          

        b1 = self.shape['y1']
        b2 = self.shape['y2']
        b3 = self.shape['y3']
        c1 = self.shape['z1']
        c2 = self.shape['z2']
        c3 = self.shape['z3']
        a_start = self.shape['x']

        self.yz = np.array([
        [a_start, b1, c1],
        [a_start, b2, c2],
        [a_start, b3, c3]
        ])

        b1 = self.shape_end['y1']
        b2 = self.shape_end['y2']
        b3 = self.shape_end['y3']
        c1 = self.shape_end['z1']
        c2 = self.shape_end['z2']
        c3 = self.shape_end['z3']
        a_end = self.shape_end['x']

        self.yz_end = np.array([
        [a_end, b1, c1],
        [a_end, b2, c2],
        [a_end, b3, c3]
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

###################################################################################################################
###################################################################################################################


class SuperStructure:
    def __init__(self):
        self.beam = None
        self.floorbeam = None
        
        
class Beam(SuperStructure):
    def __init__(self, w_beam, h_beam, l_beam, t_webplate, t_flange, h_deck, t_deck, name, tran = None, quat = None):
        super().__init__()
        
        self.beam_width = w_beam
        self.beam_height = h_beam
        self.beam_length = l_beam
        self.webplate_thickness = t_webplate
        self.flange_thickness = t_flange
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.tran = tran
        self.quat = quat

        self.name = name
    

        cfg_start = setBeamBasic(self.beam_width, self.beam_height, self.webplate_thickness, self.flange_thickness, self.deck_height, self.deck_thickness)
        cfg_end = cfg_start

        member = IBeam(cfg_start, cfg_end, self.beam_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.beam = member.obj
 
    

class Floorbeam(SuperStructure):
    def __init__(self, t_deck, h_deck, name, w_truss, t_truss, h_truss):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        
        self.name = name
        self.truss_width = w_truss
        self.truss_thickness = t_truss 
        self.truss_height = h_truss
   

        t_h = 0.1
        t_v = 0.1
        
        
        b11 = -self.truss_width/2 + t_v
        c11 = self.deck_height - self.deck_thickness - self.truss_height/2 - 2*self.truss_height/5
        b12 = -self.truss_width/2 + t_v + 2*self.truss_width/5
        c12 = self.deck_height - self.deck_thickness - self.truss_height/2
        b13 = -self.truss_width/2 + t_v
        c13 = self.deck_height - self.deck_thickness - self.truss_height/2 + 2*self.truss_height/5
        
        b21 = self.truss_width/2 - t_v
        c21 = self.deck_height - self.deck_thickness - self.truss_height/2 - 2*self.truss_height/5
        b22 = self.truss_width/2 - t_v
        c22 = self.deck_height - self.deck_thickness - self.truss_height/2 + 2*self.truss_height/5
        b23 = self.truss_width/2 - t_v - 2*self.truss_width/5
        c23 = self.deck_height - self.deck_thickness - self.truss_height/2
        
        b31 = 2*self.truss_width/5
        c31 = self.deck_height - self.deck_thickness - self.truss_height + t_h
        b32 = 0
        c32 = self.deck_height - self.deck_thickness - self.truss_height + t_h + 7*self.truss_height/20
        b33 = -2*self.truss_width/5
        c33 = self.deck_height - self.deck_thickness - self.truss_height + t_h
        
        b41 = 0
        c41 = self.deck_height - self.deck_thickness - t_h - 7*self.truss_height/20
        b42 = 2*self.truss_width/5
        c42 = self.deck_height - self.deck_thickness - t_h
        b43 = -2*self.truss_width/5
        c43 = self.deck_height - self.deck_thickness - t_h
        
        
        cfg_start = setRectangle(self.truss_width, self.truss_height, self.deck_height - self.deck_thickness)  
        cfg_end = cfg_start

        orig = Rectangle(cfg_start, cfg_end, self.truss_thickness)
        orig.createObj(self.name)
        
        name = 'left_hollow'
        cfg1_start = setTriangle(1, b11, c11, b12, c12, b13, c13) 
        cfg1_end = setTriangle(-1, b11, c11, b12, c12, b13, c13)
        tria1 = Triangle(cfg1_start, cfg1_end, 1)
        tria1.createObj(name)
        Hollow(orig.obj, tria1.obj)        

        name = 'right_hollow' 
        cfg2_start = setTriangle(1, b21, c21, b22, c22, b23, c23) 
        cfg2_end = setTriangle(-1, b21, c21, b22, c22, b23, c23)
        tria2 = Triangle(cfg2_start, cfg2_end, 1)
        tria2.createObj(name)
        Hollow(orig.obj, tria2.obj)       
       
        name = 'down_hollow' 
        cfg3_start = setTriangle(2, b31, c31, b32, c32, b33, c33) 
        cfg3_end = setTriangle(-2, b31, c31, b32, c32, b33, c33)
        tria3 = Triangle(cfg3_start, cfg3_end, 1)
        tria3.createObj(name)
        Hollow(orig.obj, tria3.obj)     
        
        name = 'up_hollow'
        cfg4_start = setTriangle(1, b41, c41, b42, c42, b43, c43) 
        cfg4_end = setTriangle(-2, b41, c41, b42, c42, b43, c43)
        tria4 = Triangle(cfg4_start, cfg4_end, 1)
        tria4.createObj(name)
        Hollow(orig.obj, tria4.obj) 


        self.floorbeam = orig.obj




    
class Deck:
    def __init__(self):
        self.slab = None
     
class Slab(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_deck, name, \
        tran = None, quat = None):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        self.deck_length = l_deck
        self.deck_width = w_deck
     
        self.truss_thickness = None

        self.tran = tran
        self.quat = quat

        self.name = name

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

class SteelGirderBridge:

    
    def __init__(self, l_deck = 500, h_deck = 11, t_deck = 1, w_deck = 10, d_column = 10, l_column = 2, w_column = 9):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column
        
        
        self.beam_height = 1.2
        self.beam_length = 500
        self.beam_width = 1
        self.flange_thickness = 0.1
        self.webplate_thickness = 0.1
        
        self.truss_width = self.deck_width/4 - self.webplate_thickness
        self.truss_height = self.beam_height - self.flange_thickness
        self.truss_thickness = 0.5
        self.truss_distance = 5
        self.truss_number = int(self.deck_length/self.truss_distance)

        self.bearing_thickness = 0.3

        self.column_height = self.deck_height - self.deck_thickness -self.truss_height - self.bearing_thickness
        self.column_number = int(self.deck_length/self.column_distance)
        self.column_thickness = 1   


        #index_deck_list = [1,4,6]
        self.deck_index = 1
        #self.deck_index = random.choice(index_deck_list)

        self.a = (np.random.rand())*20
        self.b = (np.random.rand())*1000

        self.theta = np.array([np.arctan(float(self.fyd(i,self.a,self.b))) for i in range(1,self.deck_length+1)])


        self.beam()
        self.floorbeam()
        self.deck()
        self.column()
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
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete solid deck', t, quat)
            member.concrete_solid()
        
        elif self.deck_index == 2:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete PK deck', t, quat)
            member.concrete_PK()
        
        elif self.deck_index == 3:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete box deck', t, quat)
            member.concrete_box()
        
        elif self.deck_index == 4:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete constalia deck', t, quat)
            member.concrete_costalia()

        elif self.deck_index == 5:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'steel box deck', t, quat)
            member.steel_box() 

        elif self.deck_index == 6:
            member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'steel sidebox deck', t, quat)
            member.steel_sidebox()     

    def beam(self):
        n = self.beam_length
        t = np.zeros((n,3))
        t[:,0] = np.array([self.fx(i,self.a,self.b) for i in range(n)])
        t[:,1] = np.array([self.fy(i,self.a,self.b) for i in range(n)])
        rotvec = np.zeros((n,3))
        rotvec[:,2] = self.theta
        Rot = R.from_rotvec(rotvec)
        quat = Rot.as_quat()
        
        member = Beam(self.beam_width, self.beam_height, self.beam_length, self.webplate_thickness, self.flange_thickness, self.deck_height, self.deck_thickness, 'i_beam', t, quat)

        for i in range(self.beam_length):
            
            theta = float(self.fyd(i,self.a,self.b))

        Duplicate(member.beam, 'beam1', None, 3*self.deck_width/8, None, 0,0, None)

        Duplicate(member.beam, 'beam2', None, self.deck_width/8, None, 0, 0, None)
           
        Duplicate(member.beam, 'beam3', None, -self.deck_width/8, None, 0,0, None)

        Duplicate(member.beam, 'beam4', None, -3*self.deck_width/8, None, 0, 0, None)
         

        member.beam.select_set(True)
        bpy.ops.object.delete()


    def floorbeam(self):
        
        member = Floorbeam(self.deck_thickness, self.deck_height, 'floorbeam', self.truss_width, self.truss_thickness,self.truss_height) 

        for i in range(self.truss_number):
            # dis = self.truss_thickness/2 + 0.5
            
            dis = self.deck_width/4

            theta = float(self.fyd(int(i*self.truss_distance),self.a,self.b))

            Duplicate(member.floorbeam, 'floorbeam1'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.truss_distance),self.a,self.b) + dis*np.cos(theta), None, 0, 0, np.arctan(theta))

            Duplicate(member.floorbeam, 'floorbeam2'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b),\
                self.fy(int(i*self.truss_distance),self.a,self.b), None, 0, 0, np.arctan(theta))

            Duplicate(member.floorbeam, 'floorbeam3'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.truss_distance),self.a,self.b) - dis*np.cos(theta), None, 0, 0, np.arctan(theta))

        member.floorbeam.select_set(True)
        bpy.ops.object.delete()

    def column(self):

        #self.column_height -= self.bearing_thickness
        member = Column(self.column_width, self.column_height, self.column_thickness, self.column_length,  self.deck_height, self.deck_thickness, 'pi column')
        member.piColumn()        
        for i in range(self.column_number): 

            theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
            print('theta = ' + str(theta))
            
            Duplicate(member.column, 'Pi_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(theta))            

        member.column.select_set(True)
        bpy.ops.object.delete()
        #self.column_height += self.bearing_thickness


    def bearing(self):
        a = self.beam_width
        T = self.bearing_thickness
        t = T/3
        d = a

        member = ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
            dis = a/2 + 0.5

            theta = float(self.fyd(i*self.column_distance,self.a,self.b))      
        
            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta)*2,\
                 self.fy(int(i*self.column_distance),self.a,self.b) + 3*self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))

            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) + self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))
        
            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) -self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))

            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta)*2,\
                self.fy(int(i*self.column_distance),self.a,self.b) -3*self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()

SteelGirderBridge()   