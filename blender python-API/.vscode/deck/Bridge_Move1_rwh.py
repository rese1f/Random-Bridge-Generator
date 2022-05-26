import bpy
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
import torch
import math
from sympy import *

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

#########################################################
#########################################################
################# Zone for Modification ################# 
#########################################################
bb1=5 # Width of the deck               #################
bb2=4 # Width of the columns            #################
Length_of_Bridge=1001 # Length[meter]   #################
Thickness_of_Deck=3                     #################
Thickness_of_Column=2                   #################
Span_of_Columns=20                      #################
def fz(x): ## Function of the bridge    #################
    if x <= 100:                        #################
        height=x/10                     #################
    else:                               #################
        height=10                       #################
    return height                       ################# 
def fy(x):                              #################   
    return x/10                         #################                      
#########################################################
################# Zone for Modification ################# 
#########################################################
#########################################################
Thickness_of_Column+=1    

def fyd(x):
    k=Symbol("k")
    fyd=diff(fy(k),k,1)
    fyd=fyd.subs(k,x)
    return fyd

theta=np.zeros(Length_of_Bridge)
height=np.zeros(Length_of_Bridge)
for i in range(Length_of_Bridge):
    height[i]=fz(i)
    theta[i]=np.arctan(float(fyd(i)))

class CrossSection:
    def __init__(self):
        self.C = None
        self.yz = None
    
    def Rectangule(self,b,h,h2):
        yz = torch.tensor([
            [-0.5*b,h-h2],
            [0.5*b,h-h2],
            [0.5*b,h],
            [-0.5*b,h]
        ])
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
        self.t= t
        
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

###############################################

#################### Decks ####################

###############################################

if __name__ == "__main__":
    cs = CrossSection()
    n = Length_of_Bridge
    # create constant rectangular cross section
    for i in range(n):
        hb=height[i]
        if hb <= Thickness_of_Deck:
            h2=hb
        else:
            h2=Thickness_of_Deck
        cs.Rectangule(bb1,hb,h2)
    cs.Show()
    
    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n,3))
    t[:,0] = np.arange(n)
    a = 0
    omega = np.arctan(float(fyd(i)))
    t[:,1] = np.sin(omega*np.arange(n)) * a
    rotvec = np.zeros((n,3))
    rotvec[:,1] = -np.arctan(omega * a)
    Rot = R.from_rotvec(rotvec)
    
    m = Member(cs.C,t,Rot.as_quat())
    #m.Show()
#plt.show()
vertices = m.v
edges=[ ]
faces=m.f
f1=()
f2=()
for i in range(m.npts):
    f1+=(i,)
    f2+=(m.npts*n-i-1,)
faces.append(f1)
faces.append(f2)

new_mesh=bpy.data.meshes.new("new_mesh")
new_mesh.from_pydata(vertices,edges,faces)
new_mesh.update()
deck=bpy.data.objects.new("deck",new_mesh)
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(deck)



###############################################

################### Columns ###################

###############################################


if __name__ == "__main__":
    cs = CrossSection()
    n = Thickness_of_Column
    # create constant rectangular cross section
    for i in range(n):
        hb=1
        h2=hb
        cs.Rectangule(bb2,hb,h2)
    cs.Show()
    
    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n,3)); t[:,0] = np.arange(n); 
    a = 0; omega = np.pi / (n-1)
    t[:,2] = np.sin(omega*np.arange(n)) * a
    rotvec = np.zeros((n,3))
    rotvec[:,1] = -np.arctan(omega * a * np.cos(omega*np.arange(n)))
    Rot = R.from_rotvec(rotvec)
    
    m = Member(cs.C,t,Rot.as_quat())
    #m.Show()
#plt.show()

vertices = m.v
edges=[ ]
faces=m.f

f1=()
f2=()
for i in range(m.npts):
    f1+=(i,)
    f2+=(m.npts*n-i-1,)
faces.append(f1)
faces.append(f2)

new_mesh=bpy.data.meshes.new("new_mesh")
new_mesh.from_pydata(vertices,edges,faces)
new_mesh.update()

#create an object from the mesh
column=bpy.data.objects.new("column",new_mesh)

#add the object to view
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(column)

def duplicate_distance(w,l):
    n=int(l/w)
    D=np.zeros(n+1)
    for i in range(n+1):
        D[i]=l/n*i
    return D

for i,j in zip(duplicate_distance(Span_of_Columns,Length_of_Bridge),range(len(duplicate_distance(Span_of_Columns,Length_of_Bridge)))):
    if j == 0:
        column.scale[2]=fz(i)
        column.location.x = i
        column.location.y = fy(i)
        column.rotation_euler[2] = np.arctan(float(fyd(i)))
    else:
        column = bpy.data.objects["column"]
        columnzip=bpy.data.objects.new('column%1.f' %j, column.data)
        columnzip.location.x = i
        columnzip.location.y = fy(i)
        columnzip.rotation_euler[2] = np.arctan(float(fyd(i)))
        columnzip.scale[2]=fz(i)-Thickness_of_Deck/2
        bpy.data.collections["Collection"].objects.link(columnzip)
        
    

'''
    if __name__ == "__main__":
        cs = CrossSection()
        n = Thickness_of_Column
        # create constant rectangular cross section
        k=i
        for i in range(n):
            hb=10
            h2=hb
            cs.Rectangule(bb2,hb,h2)
        cs.Show()
        
        # define arbitrary cross-sectional translation and rotation
        t = np.zeros((n,3)); t[:,0] = np.arange(n); 
        a = 0; omega = np.pi / (n-1)
        t[:,2] = np.sin(omega*np.arange(n)) * a
        rotvec = np.zeros((n,3))
        rotvec[:,1] = -np.arctan(omega * a * np.cos(omega*np.arange(n)))
        Rot = R.from_rotvec(rotvec)
        
        m = Member(cs.C,t,Rot.as_quat())
        #m.Show()
    #plt.show()
    vertices = m.v
    edges=[ ]
    faces=m.f
    f1=()
    f2=()
    for i in range(m.npts):
        f1+=(i,)
        f2+=(m.npts*n-i-1,)
    faces.append(f1)
    faces.append(f2)

    new_mesh=bpy.data.meshes.new("new_mesh")
    new_mesh.from_pydata(vertices,edges,faces)
    new_mesh.update()
    column=bpy.data.objects.new("column",new_mesh)
    view_layer=bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(column)
    column.location.x = i
    '''
    
    

