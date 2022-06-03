import bpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

Width_of_Bridge = 10 # Width of the deck                   
Length_of_Bridge = 1001 # Length[meter]   
Thickness_of_Deck = 2                     
Thickness_of_Column = 5                  

column_x1 = 200 # X - coordinate of the first column
column_x2 = 800 # X - coordinate of the second column
Span_of_Columns = column_x2-column_x1
Height_of_Column = 60
Width_of_Column = 4

Span_of_Cable = 10

def fx(x):
    return x

def fy(x):
    return 0

def fz(x):
    return 10
'''
The track of Bridge, three dimensions
'''


def fyd(x):
    '''
    derivative of fy; Useless in this file
    '''
    k = Symbol("k")
    fyd = diff(fy(k),k,1)
    fyd = fyd.subs(k,x)
    return fyd



def fz_cable1(x):
    return ((0.9*Height_of_Column-fz(x))/column_x1**2)*x**2+fz(x)

def fz_cable2(x):
    return (4*(Height_of_Column*0.9-fz(x)-6)/Span_of_Columns**2)*(x-Span_of_Columns/2)**2+fz(x)+6

def fz_cable3(x):
    return ((0.9*Height_of_Column-fz(x))/(Length_of_Bridge - column_x2)**2)*(x-(Length_of_Bridge - column_x2))**2+fz(x)
'''
The function of the Main Cable
All of three functions start from 0
'''

def duplicateDistance(w,l):
    '''
    Args:
        w - Span between two columns
        l - Length of the Bridge
    Return:
        Coordinates for each column on x - axis
    '''
    n = int(l/w)
    D = np.zeros(n+1)
    for i in range(n+1):
        D[i] = l/n*i
    return D

height = np.array([fz(i) for i in range(Length_of_Bridge)])
theta = np.array([np.arctan(float(fyd(i))) for i in range(1,Length_of_Bridge)])
theta = np.concatenate((np.array([0]),theta),axis=0)

class CrossSection:
    def __init__(self):
        self.C = None
        self.yz = None

    def Rectangule(self,b,h,h2):
        '''
        b - width
        h - height on the top of deck or the height of the columns
        h2 - thickness of the deck
        '''
        yz = np.array([
            [-0.5*b,h-h2],
            [0.5*b,h-h2],
            [0.5*b,h],
            [-0.5*b,h]
        ])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def circle(self,radius_circle):
        '''
        radius_circle - generate a circle
        '''
        alpha = np.linspace(0, 2*np.pi, 50)
        x = np.array(np.cos(alpha))*radius_circle
        y = np.array(np.sin(alpha))*radius_circle
        yz = np.zeros((len(x),2))
        yz[:,0] = x
        yz[:,1] = y
        yz = np.delete(yz,-1,axis=0).round(2)
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def AddCurrentSection(self):
        yz = self.yz.reshape((1,self.yz.shape[0],self.yz.shape[1]))
        if self.C is None:
            self.C = yz
        else:
            self.C = np.concatenate((self.C,yz),0)
        
    def Show(self):
        idx = np.mod(np.arange(self.yz.shape[0]+1),self.yz.shape[0])
        plt.figure()
        plt.plot(self.yz[idx,0],self.yz[idx,1])
        plt.xlabel('y'); plt.ylabel('z')
        plt.axis('equal')
        plt.title("Cross-section {:d}, {:d} points".format(self.C.shape[0],self.C.shape[1]))

class Member:
    def __init__(self,C,t=None,quat=None):
        # t: translation of the cross-sections
        # quat: rotation (quaternion) of the cross-sections
        self.n = C.shape[0]
        self.npts = C.shape[1]
        if t is None:
            t = np.zeros((self.n,3))
            t[:,0] = np.arange(self.n)
        self.t = t
        
        if quat is None:
            quat = np.zeros((self.n,4))
            quat[:,3] = 1. #quaternion is (x,y,z,w)
        self.R = R.from_quat(quat)   
        
        self.v = []
        self.f = []
        
        xyz = np.zeros((self.npts,3))
        for i in range(self.n):
            xyz[:,1:] = C[i,:,:]
            c = self.R[i].apply(xyz) + self.t[np.zeros(self.npts,dtype=int)+i,:]
            self.v = self.v + [(c[k,0],c[k,1],c[k,2]) for k in range(self.npts)]
            if i>0:
                m = self.npts * (i-1)
                idx1 = np.arange(m,m+self.npts)
                idx2 = np.arange(m+self.npts,m+2*self.npts)
                self.f = self.f + [(idx1[k],idx1[np.mod(k+1,self.npts)],idx2[np.mod(k+1,self.npts)],idx2[k]) for k in range(self.npts)]
        
    def Show(self):
        v = np.array(self.v)
        plt.figure()
        ax = plt.axes(projection='3d')
        for f in self.f:
            idxv = np.array(f)
            ax.plot3D(v[idxv,0], v[idxv,1], v[idxv,2], 'k')
            
        vmin = v.min(0); vmax = v.max(0); ctr = (vmin+vmax)/2.
        half = (vmax-vmin).max()/2.
        ax.set_xlim((ctr[0]-half,ctr[0]+half))
        ax.set_ylim((ctr[1]-half,ctr[1]+half))
        ax.set_zlim((ctr[2]-half,ctr[2]+half))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

def slabBuild(n,thickness_deck,height,width,name):
    '''
    Args:
        n - length of the bridge
        thickness_deck - thickness of the deck (z direction)
        width - width of the deck
        name - the name of components in Blender
    Return:
        Deck Model
    '''
    cs = CrossSection()
    for i in range(n):
        hb = height[i]
        h2 = min(thickness_deck,hb)
        cs.Rectangule(width,hb,h2)
    cs.Show()
    t = np.zeros((n,3))
    t[:,0] = np.array([fx(i) for i in range(n)])
    t[:,1] = np.array([fy(i) for i in range(n)])
    rotvec = np.zeros((n,3))
    rotvec[:,2] = theta
    Rot = R.from_rotvec(rotvec)
    m = Member(cs.C,t,Rot.as_quat())
    vertices = m.v
    edges = [ ]
    faces = m.f
    
    f1 = ()
    f2 = ()
    for i in range(m.npts):
        f1 += (i,)
        f2 += (m.npts*n-i-1,)
    faces.append(f1)
    faces.append(f2)

    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices,edges,faces)
    new_mesh.update()
    deckzip = bpy.data.objects.new(name,new_mesh)
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(deckzip)
    return

def mainColumnBuild(width,height,x_dis,y_dis,name):
    '''
    Args:
        Width - Width of Column
        Height - Height of Column
        x_dis - x coordinate of Column
        y_dis - y coordinate of Column
    Returns:
        One Column Model
    '''
    cs = CrossSection()
    n = Thickness_of_Column
    for i in range(n):
        h2=height
        cs.Rectangule(width,height,h2)
    m = Member(cs.C,None,None)
    vertices = m.v
    edges = [ ]
    faces = m.f
    f1 = ()
    f2 = ()
    for i in range(m.npts):
        f1 += (i,)
        f2 += (m.npts*n-i-1,)
    faces.append(f1)
    faces.append(f2)
    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices,edges,faces)
    new_mesh.update()
    main_column_1 = bpy.data.objects.new(name,new_mesh)
    main_column_1.location.x = x_dis
    main_column_1.location.y = y_dis
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(main_column_1)

def mainCableBuild(Length,radius,x_dis,y_dis,function,name):
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
    cs=CrossSection()
    n = Length
    for i in range(n):
        cs.circle(radius)
    t = np.zeros((n,3))
    t[:,0] = np.array([fx(i) for i in range(n)])
    t[:,1] = np.array([fy(i) for i in range(n)])
    t[:,2] = np.array([function(i) for i in range(n)])
    #rotvec = np.zeros((n,3))
    #rotvec[:,2] = theta
    #Rot = R.from_rotvec(rotvec)
    
    m = Member(cs.C,t,None)
    vertices = m.v
    edges = [ ]
    faces = m.f

    f1 = ()
    f2 = ()
    for i in range(m.npts):
        f1 += (i,)
        f2 += (m.npts*n-i-1,)
    faces.append(f1)
    faces.append(f2)
    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices,edges,faces)
    new_mesh.update()
    main_cable_1 = bpy.data.objects.new(name,new_mesh)
    main_cable_1.location.x = x_dis
    main_cable_1.location.y = y_dis
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(main_cable_1)
    return

def cableBuild(Span,Length,radius,x_dis,y_dis,function,name):
    '''
    Args:
        Span - Span distance between two cables
        Length - Length of this part of main cable
        radius - radius of main cable (default r = 0.1)
        x_dis - x coordinate of Start of Main Cable
        y_dis - y coordinate of Start of Main Cable
        function - the mathmatical function of main cable in particuler part
    Returns:
        One Column Model
    '''
    cs = CrossSection()
    n=2 # Set the length of cable as 1 first, then use [scale] to change the length
    for i in range(n):
        cs.circle(radius)
    m = Member(cs.C,None,None)
    vertices = m.v
    edges = [ ]
    faces = m.f

    f1 = ()
    f2 = ()
    for i in range(m.npts):
        f1 += (i,)
        f2 += (m.npts*n-i-1,)
    faces.append(f1)
    faces.append(f2)

    new_mesh = bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices,edges,faces)
    new_mesh.update()
    cable = bpy.data.objects.new(name,new_mesh)
    cable.location.y = y_dis
    cable.rotation_euler[1] = -np.pi/2
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(cable)

    for j,i in enumerate(duplicateDistance(Span,Length)):    
        if j == 0:
            cable.scale[0] = 0
            cable.location.x = x_dis+i
            cable.location.y = y_dis
            cable.location.z = fz(i)
        else:
            cable = bpy.data.objects[name]
            namezip = str(name) + str(j)
            cablezip = bpy.data.objects.new(namezip,cable.data)
            cablezip.location.x = x_dis+i
            cablezip.location.y = y_dis
            cablezip.location.z = fz(i)
            cablezip.rotation_euler[1] = - np.pi/2 
            cablezip.scale[0] = function(i) - fz(i)
            bpy.data.collections["Collection"].objects.link(cablezip)
    return


slabBuild(Length_of_Bridge,Thickness_of_Deck,height,Width_of_Bridge,"Slab")

mainColumnBuild(Width_of_Column,Height_of_Column,column_x1, Width_of_Bridge/2,'MainColumn1')
mainColumnBuild(Width_of_Column,Height_of_Column,column_x1,-Width_of_Bridge/2,'MainColumn2')
mainColumnBuild(Width_of_Column,Height_of_Column,column_x2, Width_of_Bridge/2,'MainColumn3')
mainColumnBuild(Width_of_Column,Height_of_Column,column_x2,-Width_of_Bridge/2,'MainColumn4')

mainCableBuild(Span_of_Columns,0.2,column_x1, Width_of_Bridge/2,fz_cable2,"MainCableMiddle1")
mainCableBuild(Span_of_Columns,0.2,column_x1,-Width_of_Bridge/2,fz_cable2,"MainCableMiddle2")
mainCableBuild(column_x1,0.2,0, Width_of_Bridge/2,fz_cable1,"MainCableLeft1")
mainCableBuild(column_x1,0.2,0,-Width_of_Bridge/2,fz_cable1,"MainCableLeft2")
mainCableBuild(Length_of_Bridge-column_x2,0.2,column_x2, Width_of_Bridge/2,fz_cable3,"MainCableRight1")
mainCableBuild(Length_of_Bridge-column_x2,0.2,column_x2,-Width_of_Bridge/2,fz_cable3,"MainCableRight2")

cableBuild(Span_of_Cable,Span_of_Columns,0.1,column_x1, Width_of_Bridge/2,fz_cable2,"cableMiddle1")
cableBuild(Span_of_Cable,Span_of_Columns,0.1,column_x1,-Width_of_Bridge/2,fz_cable2,"cableMiddle2")
cableBuild(Span_of_Cable,column_x1,0.1,0, Width_of_Bridge/2,fz_cable1,"cableLeft1")
cableBuild(Span_of_Cable,column_x1,0.1,0,-Width_of_Bridge/2,fz_cable1,"cableLeft2")
cableBuild(Span_of_Cable,Length_of_Bridge-column_x2,0.1,column_x2, Width_of_Bridge/2,fz_cable3,"cableRight1")
cableBuild(Span_of_Cable,Length_of_Bridge-column_x2,0.1,column_x2,-Width_of_Bridge/2,fz_cable3,"cableRight2")