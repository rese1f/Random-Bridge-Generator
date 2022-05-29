# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 09:14:17 2022

@author: naraz
"""

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class CrossSection:
    # define cross section in the local coordinate system
    # x is the direction perpendicular to the cross-section
    # cross-sections are defined in yz plane
    def __init__(self):
        self.C = None
        self.yz = None
        
    def Rectangule(self,b,h):
        yz = np.zeros((4,2)) #four row, two column
        yz[:,0] = np.array([-0.5*b,0.5*b,0.5*b,-0.5*b])
        yz[:,1] = np.array([-0.5*h,-0.5*h,0.5*h,0.5*h])
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
    n = 10 #face
    # create constant rectangular cross section
    for i in range(n):
        cs.Rectangule(1.0, 2.0)
    cs.Show()
    
    # define arbitrary cross-sectional translation and rotation
    t = np.zeros((n,3)); t[:,0] = np.arange(n); 
    a = 1.5; omega = np.pi / (n-1)
    t[:,2] = np.sin(omega*np.arange(n)) * a
    rotvec = np.zeros((n,3))
    rotvec[:,1] = -np.arctan(omega * a * np.cos(omega*np.arange(n)))
    Rot = R.from_rotvec(rotvec)
    
    m = Member(cs.C,t,Rot.as_quat())
    #m = Member(cs.C,None,None)
    m.Show()
plt.show()