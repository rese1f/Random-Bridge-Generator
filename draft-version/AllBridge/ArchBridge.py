# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:37:28

@author: dell
"""
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



def setArchringBasic(w_archring, t_archring, h_archring):
    cfg = {
        "name": "archring_basic",
        "shape": {
            'archring width': w_archring,
            'archring thickness': t_archring,
            'archring top surface height': h_archring
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
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
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
            [(w_deck / 2 - m), (h_deck)],
            [(w_deck / 2), (h_deck + t_deck / 2)],
            [(w_deck / 2 - m), (h_deck + t_deck)],
            [-(w_deck / 2 - m), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck / 2)],
            [-(w_deck / 2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()


class ConcretePK(Member):
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck / 8, w_deck / 4)
        self.yz = np.array([
            [(w_deck / 2 - m), (h_deck)],
            [(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()


class ConcreteBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck / 8, w_deck / 4)
        m2 = m / 2
        self.yz = np.array([
            [(w_deck / 2 - m), (h_deck)],
            [(w_deck / 2 - m2), (h_deck + t_deck - 0.2)],
            [(w_deck / 2), (h_deck + t_deck - 0.2)],
            [(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck - 0.2)],
            [-(w_deck / 2 - m2), (h_deck + t_deck - 0.2)],
            [-(w_deck / 2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()


class ConcreteCostalia(Member):
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
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
            [(w_deck / 2), (h_deck)],
            [(w_deck / 2 - m), (h_deck + t_deck)],
            [-(w_deck / 2 - m), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()


class SteelBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        m = random.uniform(w_deck / 6, w_deck / 4)
        m2 = random.uniform(w_deck / 18, w_deck / 12)

        self.yz = np.array([
            [(w_deck / 2 - m), (h_deck)],
            [(w_deck / 2), (h_deck + t_deck / 2)],
            [(w_deck / 2 - m2), (h_deck + t_deck)],
            [-(w_deck / 2 - m2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck / 2)],
            [-(w_deck / 2 - m), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()


class SteelSidebox(Member):
    def __init__(self, cfg, cfg_end, n, three_d=False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_deck = self.shape['deck width']
        t_deck = self.shape['deck thickness']
        h_deck_t = self.shape['deck top surface height']
        h_deck = h_deck_t - t_deck

        h_deck = h_deck_t - t_deck  # height at bottom of deck

        self.yz = np.array([
            [(w_deck / 2), (h_deck)],
            [(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck + t_deck)],
            [-(w_deck / 2), (h_deck)]
        ])

        self.yz_end = self.yz

        self.setMember3d()





class ArchRing(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']

        w_archring = self.shape['archring width']
        t_archring = self.shape['archring thickness']
        h_archring_t = self.shape['archring top surface height']
        #h_archring = h_archring_t - t_archring

        h_archring = h_archring_t - t_archring # height at bottom of archring
        
        self.yz = np.array([
            [(w_archring/2), (h_archring)],
            [(w_archring/2), (h_archring + t_archring)],
            [-(w_archring/2), (h_archring + t_archring)],
            [-(w_archring/2), (h_archring)]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()          




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

class Deck:
    def __init__(self):
            self.slab = None

class Slab(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_deck, name):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        self.deck_length = l_deck
        self.deck_width = w_deck
        self.name = name

    def concrete_solid(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteSolid(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj

    def concrete_PK(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcretePK(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj

    def concrete_box(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteBox(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj

    def concrete_costalia(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = ConcreteCostalia(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj

    def steel_box(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = SteelBox(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj

    def steel_sidebox(self):
        cfg_start = setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = SteelSidebox(cfg_start, cfg_end, self.deck_length)
        member.createObj(self.name)
        self.slab = member.obj





class SuperStructure:
    def __init__(self):
        self.archring = None
        self.uprightcolumn = None

class Archring(SuperStructure):
    def __init__(self, t_archring, h_archring, l_archring, w_archring, name, t=None, quat=None):
        super().__init__()

        self.archring_thickness = t_archring
        self.archring_height = h_archring
        self.archring_length = l_archring
        self.archring_width = w_archring
        self.name = name
        
        self.t = t
        self.quat = quat

        cfg_start = setArchringBasic(self.archring_width, self.archring_height, self.archring_thickness)
        cfg_end = cfg_start


        member = ArchRing(cfg_start, cfg_end, self.archring_length, False, self.t, self.quat)
        member.createObj(self.name)
        self.archring = member.obj

# a = Archring(1, 0.5, 300, 2, 'arch') 
# arch = a.archring
# arch.location.z = 2       


class Uprightcolmn(SuperStructure):
    def __init__(self, w_column, h_column, t_column, l_column, h_deck_t, t_deck, name):
        super().__init__()
        self.column_width = w_column
        self.column_height = h_column
        self.column_thickness = t_column # latitude
        self.column_length = l_column # longitude
        self.deck_height = h_deck_t
        self.deck_thickness = t_deck
        self.name = name

  
        cfg_start = setRectangle(self.column_width, self.column_height, self.column_height)
        cfg_end = cfg_start

   
        member = Rectangle(cfg_start, cfg_end, self.column_length)
        member.createObj(self.name)
        self.uprightcolumn = member.obj              
                
                
    def duplicateDistance(self, w, l):
        n = int(l / w)
        D = np.zeros(n)
        for i in range(n):
            D[i] = w / 2 + w * i
        return D


class Bearing:
    def __init__(self):
        self.column_bearing = None


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



class SubStructure:
    def __init__(self):
        self.pier = None


class Pier(SubStructure):
    def __init__(self, w_pier, h_pier, t_pier, name):
        super().__init__()
        self.pier_width = w_pier
        self.pier_height = h_pier
        self.pier_thickness = t_pier 
        self.name = name

  
        cfg_start = setRectangle(self.pier_width, self.pier_height, self.pier_height)
        cfg_end = cfg_start

   
        member = Rectangle(cfg_start, cfg_end, self.pier_thickness)
        member.createObj(self.name)
        self.pier = member.obj              
                
        
#################################################################################################################
#################################################################################################################

class DeckArchBridge:

    def __init__(self,w_deck = 10, h_deck = 30, t_deck = 3, l_deck = 102, l_column = 2, w_column = 2):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_length = l_column
        self.column_width = w_column
        self.archring_height = 2
        self.archring_length = 100
        self.archring_thickness = 1
        self.archring_width = 10
        self.archring_heightestpoint = 25
        self.column_span = 10
        self.column_number = int(self.deck_length/self.column_span)+1 
        self.column_thickness = 1
        
        self.deck_index = 3
 

        self.theta = np.zeros([1,self.archring_length])


        self.bearing_thickness = 0.5
        
        self.pier_width = self.archring_width + 2
        self.pier_height = 4
        self.pier_thickness = 6




        self.deck()
        self.archring()
        self.uprightcolumn()
        self.bearing()
        self.pier()



        
    def deck(self):
        member = Slab(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'Deck')
        if self.deck_index == 1:
            member.concrete_solid()
        elif self.deck_index == 2:
            member.concrete_PK()
        elif self.deck_index == 3:
            member.concrete_box()
        elif self.deck_index == 4:
            member.concrete_costalia()
        elif self.deck_index == 5:
            member.steel_box()
        elif self.deck_index == 6:
            member.steel_sidebox()
            
        member.slab.location.x = self.archring_length/2



    def fx(self,x):
        return x

    def fy(self,x):
        return 0

#    def fyd(self,x):
#        fyd = (self.fy(x+1e-9,self.a,self.b)-self.fy(x,self.a,self.b))/1e-9
#        return fyd

    def fz(self,x):
        return (-4*self.archring_heightestpoint/self.archring_length/self.archring_length)*x**2+4*self.archring_heightestpoint/self.archring_length*x + 1  # highest point(l/2,h1)  
    
    def archring(self):
        n = self.archring_length
        t = np.zeros((n,3))
        t[:,0] = np.array([self.fx(i) for i in range(n)])
        t[:,2] = np.array([self.fz(i) for i in range(n)])
        rotvec = np.zeros((n,3))
        rotvec[:,2] = self.theta
        Rot = R.from_rotvec(rotvec)
        quat = Rot.as_quat()
    
        member = Archring(self.archring_thickness, self.archring_height, self.archring_length, self.archring_width, 'Archring', t, quat)
        
        
        
    def uprightcolumn(self):
        n = self.column_number
        for i in range (n):
            self.column_height = 1
            
        member = Uprightcolmn(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'Uprightcolumn')
        
        
        for i in range (n):
            
            Duplicate(member.uprightcolumn, 'uprightcolumn'+str(i+1), self.fx(int(i*self.column_span)), self.fy(int(i*self.column_span)) + 1*self.deck_width/4,\
                        self.fz(int(i*self.column_span)), 0, 0, 0, None, None, self.deck_height-self.deck_thickness-self.bearing_thickness-self.fz(i*self.column_span))
       
            Duplicate(member.uprightcolumn, 'uprightcolumn'+str(i+1), self.fx(int(i*self.column_span)), self.fy(int(i*self.column_span)) - 1*self.deck_width/4,\
                        self.fz(int(i*self.column_span)), 0, 0, 0, None, None, self.deck_height-self.deck_thickness-self.bearing_thickness-self.fz(i*self.column_span))
       
        
        member.uprightcolumn.select_set(True)
        bpy.ops.object.delete()


    def bearing(self):
        a = self.column_length
        T = self.bearing_thickness
        t = T/3
        d = a

        member = ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
        
            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_span)),\
                self.fy(int(i*self.column_span)) + 1*self.deck_width/4, self.deck_height - self.bearing_thickness/2 - self.deck_thickness,0,math.pi/2,0, None, None, 0.5)

            Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_span)),\
                self.fy(int(i*self.column_span)) - 1*self.deck_width/4, self.deck_height - self.bearing_thickness/2 - self.deck_thickness,0,math.pi/2,0, None, None, 0.5)
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()
       
        
    def pier(self):
        
        member = Pier(self.pier_width, self.pier_height, self.pier_thickness, "pier")

        Duplicate(member.pier, 'pier', self.deck_length - self.pier_thickness/3, None, None,0, 0, 0, None, None, 1)

DeckArchBridge()          

          