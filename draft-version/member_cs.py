# import bpy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *
import math

# cf = bpy.data.texts["cfg.py"].as_module()
import cfg as cf

class Member:
    """
    define cross section in the local coordinate system
    x is the direction perpendicular to the cross-section
    cross-sections are defined in yz plane
    """

    def __init__(self, cfg, n=1, t=None, quat=None, type = None):
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
        self.type = type

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
        
        # print(self.quat)

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
        plt.plot(self.yz[idx, 0], self.yz[idx, 1], color = 'k')

        plt.xlabel('y')
        plt.ylabel('z')
        plt.axis('equal')
        plt.title("Cross-section of {}".format(self.type))
        plt.axis('off')
        # plt.show()

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


class ConcreteSolid(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Concrete Solid slab'):
        super().__init__(cfg, n, t, quat, type)

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
        
        self.showCrossSection()

class ConcretePK(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Concrete PK slab'):
        super().__init__(cfg, n, t, quat, type)

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
        
        self.showCrossSection()

class ConcreteBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Concrete Box slab'):
        super().__init__(cfg, n, t, quat, type)
        
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

        self.showCrossSection()      

class ConcreteCostalia(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type ='Concrete Costalia slab'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection()   


class SteelBox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Steel Box slab'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection()   

class SteelSidebox(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Steel Side Box slab'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection()   
            
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

        self.showCrossSection()   

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

        self.showCrossSection()         

class Rectangle(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)

        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']        

        w = self.shape['bottom width']
        h = self.shape['height']
        H = self.shape['height of top']

        self.yz = np.array([
            [w/2, H],
            [w/2, H-h],
            [-w/2, H-h],
            [-w/2, H]
        ])  

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

        self.showCrossSection()             

class PierCap(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)
        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']  

        w = self.shape['bottom width']
        h = self.shape['height']
        a = self.shape['a']
        b = self.shape['b']
        c = self.shape['c']

        self.yz = np.array([
            [-a-w/2,h+b],
            [-w/2,h+b],
            [-w/2,h],
            [w/2,h],
            [w/2,h+b],
            [a+w/2,h+b],
            [a+w/2,h],
            [w/2,h-c],
            [-w/2,h-c],
            [-a-w/2,h]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

        self.showCrossSection()             

class BoxPierCap(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)
        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']  

        w1 = self.shape['high width']
        w2 = self.shape['mid width']
        w3 = self.shape['low width']
        h = self.shape['height']
        a = self.shape['a']
        b = self.shape['b']

        self.yz = np.array([
            [w1/2,h],
            [w2/2,h-a],
            [w3/2,h-a-b],
            [-w3/2,h-a-b],
            [-w2/2,h-a],
            [-w1/2,h]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

        self.showCrossSection()             

class I_Beam(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None):
        super().__init__(cfg, n, t, quat)
        self.three_d = three_d
        self.cfg_end = cfg_end
        self.shape_end = self.cfg_end['shape']  

        w = self.shape['width']
        th = self.shape['thickness']
        h = self.shape['height']
        a = self.shape['a']
        b = self.shape['b']


        self.yz = np.array([
            [w/2,h],
            [w/2,h-a],
            [b/2,h-a],
            [b/2,h-th+a],
            [w/2,h-th+a],
            [w/2,h-th],
            [-w/2,h-th],
            [-w/2,h-th+a],
            [-b/2,h-th+a],
            [-b/2,h-a],
            [-w/2,h-a],
            [-w/2,h]
        ])

        self.yz_end = self.yz

        if (t is None) & (quat is None):
            self.setMember3d()
        
        else:
            self.setMember()

        self.showCrossSection()             

class A1(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'A1 Column'):
        super().__init__(cfg, n, t, quat, type) 

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

        self.showCrossSection() 

class Double(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Double Column'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection() 

class Door(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Door Column'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection()         

class Tower(Member):
    def __init__(self, cfg, cfg_end, n, three_d = False, t=None, quat=None, type = 'Tower Column'):
        super().__init__(cfg, n, t, quat, type)

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

        self.showCrossSection()         

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

        self.showCrossSection()         

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

        self.showCrossSection()             

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

        self.showCrossSection()         

class Triangle2(Member):
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

        self.showCrossSection()         

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

        self.showCrossSection()         

########################################################### plot #####################################################################

class CrossSectionPlot(Member):
    def __init__(self, component_name, cfg = None, n = 2, three_d = False, t = None, quat = None):    
    # def __init__(self, component_name):
        # super().__init__(cfg, n, t, quat, component_name)    
        self.type = component_name
        self.cfg_end = None
        self.n = n
        self.deck_width = 5
        self.deck_thickness = 0.5
        self.deck_height = 4
        self.column_width = 5
        self.column_height = 16
        self.column_thickness = 0.5 # latitude
    
    def slab(self):
        # deck_width = 4
        # deck_thickness = 2
        # # deck_length = 2
        # deck_height = deck_thickness
        self.cfg = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        self.cfg_end = self.cfg 

        if self.type == 'concrete_solid':
            ConcreteSolid(self.cfg, self.cfg_end, self.n) 
            plt.show() 

        elif self.type == 'concrete_PK':    
            ConcretePK(self.cfg, self.cfg_end, self.n) 
            plt.show()  

        elif self.type == 'concrete_box':
            ConcreteBox(self.cfg, self.cfg_end, self.n)
            plt.show() 
        
        elif self.type == 'concrete_costalia':
            ConcreteCostalia(self.cfg, self.cfg_end, self.n)
            plt.show() 
        
        elif self.type == 'steel_box':
            SteelBox(self.cfg, self.cfg_end, self.n)
            plt.show() 
        
        elif self.type == 'steel_sidebox':
            SteelSidebox(self.cfg, self.cfg_end, self.n)
            plt.show() 
    
    def column(self):
        if self.type == 'A1':
            self.cfg = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
            self.cfg_end = self.cfg

            h_deck = self.deck_height - self.deck_thickness

            k = self.column_height / (self.column_width/2)
            b_out = self.column_height
            b_in = self.column_height - k * self.column_thickness

            w_tria = -2 * (h_deck - b_in) / k
            h_tria = self.column_height - self.column_thickness * k - h_deck
            H_tria = h_deck + h_tria  

            cfg_hollow_start = cf.setTriangle2(w_tria, h_tria, H_tria)
            cfg_hollow_end = cfg_hollow_start

            # Triangle2(cfg_hollow_start, cfg_hollow_end, self.n)
            A1(self.cfg, self.cfg_end, self.n)

            H_tria = H_tria - h_tria

            yz = np.array([
                [(w_tria/2), (H_tria)],
                [0, (H_tria+h_tria)],
                [-(w_tria/2), (H_tria)],
                [(w_tria/2), (H_tria)]
            ])
            plt.plot(yz[:, 0], yz[:, 1], color = 'k')               

            plt.show()

        elif self.type == 'double':
            self.cfg = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
            self.cfg_end = self.cfg

            h_deck = self.deck_height - self.deck_thickness
            
            Double(self.cfg, self.cfg, self.n)
            
            plt.show()     

        elif self.type == 'door':
            self.cfg = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
            self.cfg_end = self.cfg

            h_deck = self.deck_height - self.deck_thickness     

            w_rec = self.column_width - 2 * self.column_thickness
            h_rec = self.column_height - 3 * self.column_thickness - h_deck
            H_rec = h_deck + h_rec

            cfg_hollow_start = cf.setRectangle(w_rec, h_rec, H_rec)
            cfg_hollow_end = cfg_hollow_start

            h_deck = self.deck_height - self.deck_thickness

            Door(self.cfg, self.cfg_end, self.n)
            
            yz = np.array([
                [w_rec/2, H_rec],
                [w_rec/2, H_rec-h_rec],
                [-w_rec/2, H_rec-h_rec],
                [-w_rec/2, H_rec],
                [w_rec/2, H_rec]
            ])  
            plt.plot(yz[:, 0], yz[:, 1], color = 'k')

            plt.show()  

        elif self.type == 'tower':
            a = random.uniform(1/5, 1/3)
            h_cable = a * self.column_height
            b = random.uniform(0.5, 0.8)
            w_bottom = b * self.column_width
            w_top = 2 * self.column_thickness

            self.cfg = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
            self.cfg_end = self.cfg

            h_deck = self.deck_height - self.deck_thickness              
            k = (self.column_height - h_cable - h_deck) / (self.column_width/2 - self.column_thickness) 
            b_cable = 0
            b_in = self.column_height - h_cable

            w_tria = self.column_width - 2 * self.column_thickness
            h_tria = self.column_height - h_cable - h_deck
            H_tria = h_deck + h_tria
            cfg_hollow_start = cf.setTriangle2(w_tria, h_tria, H_tria)
            cfg_hollow_end = cfg_hollow_start
            
            Tower(self.cfg, self.cfg_end, self.n)

            H_tria = H_tria - h_tria

            yz = np.array([
                [(w_tria/2), (H_tria)],
                [0, (H_tria+h_tria)],
                [-(w_tria/2), (H_tria)],
                [(w_tria/2), (H_tria)]
            ])
            plt.plot(yz[:, 0], yz[:, 1], color = 'k')                                                         

            plt.show()


        


a = CrossSectionPlot('concrete_solid')
a.slab()
a = CrossSectionPlot('concrete_PK')
a.slab()
a = CrossSectionPlot('concrete_box')
a.slab()
a = CrossSectionPlot('concrete_costalia')
a.slab()
a = CrossSectionPlot('steel_box')
a.slab()
a = CrossSectionPlot('steel_sidebox')
a.slab()

b = CrossSectionPlot('A1')
b.column()
b = CrossSectionPlot('double')
b.column()
b = CrossSectionPlot('door')
b.column()
b = CrossSectionPlot('tower')
b.column()

