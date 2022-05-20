# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:14:17 2022

@author: naraz
"""
# Blender Python Deck
import bpy
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

class CrossSection:
    # define cross section in the local coordinate system
    # x is the direction perpendicular to the cross-section
    # cross-sections are defined in yz plane
    def __init__(self):
        self.C = None
        self.yz = None
        
    def Rectangule(self,b,h):
        yz = np.zeros((4,2))
        yz[:,0] = np.array([-0.5*b,0.5*b,0.5*b,-0.5*b])
        yz[:,1] = np.array([-0.5*h,-0.5*h,0.5*h,0.5*h])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def I(self,b,d,th):
        yz = np.array([
            [-0.5*b ,0.5*d],
            [ 0.5*b ,0.5*d],
            [ 0.5*b ,0.5*d-th],
            [ 0.5*th,0.5*d-th],
            [ 0.5*th,-0.5*d+th],
            [ 0.5*b ,-0.5*d+th],
            [ 0.5*b ,-0.5*d],
            [-0.5*b ,-0.5*d],
            [-0.5*b ,-0.5*d+th],
            [-0.5*th,-0.5*d+th],
            [-0.5*th, 0.5*d-th],
            [-0.5*b, 0.5*d-th],
        ])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def Deck(self,b,c,d,h,e,f,a=50):
        x=np.linspace(-5,5,100)
        y=1./(1.+np.exp(-x))
        x=x*b/10+b/2+a
        y=y*c+h
        yz=np.array([
            [a+b,h+c+d+e],
            [a+b-f,h+c+d+e],
            [a+b-f,h+c+d],
            [-(a+b-f),h+c+d],
            [-(a+b-f),h+c+d+e],
            [-a-b,h+c+d+e],
        ])
        s=np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        yz=np.concatenate((s,yz),axis=0)
        s[:,0]=-s[:,0]
        s=s[::-1]
        yz=np.concatenate((yz,s),axis=0)
        yz=np.concatenate((np.array([[50,h]]),yz),axis=0)
        yz=np.concatenate((yz,np.array([[-50,h]])),axis=0)
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def Column(self,a,b,c,d,e,f):
        yz=np.array([
            [a,d],
            [a,0],
            [a+b,0],
            [a+b,d],
            [a+b+c,d+e],
            [a+b+c,d+e+f],
            [-(a+b+c),d+e+f],
            [-(a+b+c),d+e],
            [-(a+b),d],
            [-(a+b),0],
            [-a,0],
            [-a,d]
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

if __name__ == "__main__":
    cs = CrossSection()
    n = 1015
    # create constant rectangular cross section
    for i in range(n):
        #cs.Rectangule(50.0, 5.0)
        cs.Deck(h=-5,a=50,b=50,c=30,d=10,e=20,f=15)
    cs.Show()
    
    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n,3)); t[:,0] = np.arange(n); 
    a = 0; omega = np.pi / (n-1)
    t[:,2] = np.sin(omega*np.arange(n)) * a
    rotvec = np.zeros((n,3));
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
deck=bpy.data.objects.new("deck",new_mesh)

#add the object to view
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(deck)

#mod_skin=new_object.modifiers.new('Skin','SKIN')


# Move the Object
deck.location[2] = 100 # [0,1,2] for x,y,z respectively


#############################################

#### Next is to built the square columns ####

#############################################

if __name__ == "__main__":
    cs = CrossSection()
    n = 15
    # create constant rectangular cross section
    for i in range(n):
        #cs.Rectangule(50.0, 5.0)
        cs.Column(17,15,18,70,12,13)
    cs.Show()
    
    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n,3)); t[:,0] = np.arange(n); 
    a = 0; omega = np.pi / (n-1)
    t[:,2] = np.sin(omega*np.arange(n)) * a
    rotvec = np.zeros((n,3));
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

### Duplicate the column
def duplicate_distance(n,w,l):
    D=np.zeros(n+1)
    for i in range(n+1):
        D[i]=l/w*i
    return D

for i,j in zip(duplicate_distance(10,10,1000),range(len(duplicate_distance(10,10,1000)))):
    if j == 0:
        continue
    else:
        column = bpy.data.objects["column"]
        columnzip=bpy.data.objects.new('column%1.f' %j, column.data)
        columnzip.location.x = i
        bpy.data.collections["Collection"].objects.link(columnzip)
    

