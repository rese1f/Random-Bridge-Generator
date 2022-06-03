import bpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

            

# column_x1 = 200 # X - coordinate of the first column
# column_x2 = 800 # X - coordinate of the second column
# Span_of_Columns = column_x2-column_x1
# Height_of_Column = 60
# Width_of_Column = 4

# Span_of_Cable = 10

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



class CrossSection:
    def __init__(self):
        self.C = None
        self.yz = None

    def Rectangular_deck(self,b,h,h2):
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
    
    def real_deck(self,b,c,d,e,f,a=50,h=95):
        """
        @rwh

        """
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
        self.addsection()
        return yz


    def circle_cable(self,radius_circle):
        '''
        radius_circle - generate a circle
        '''
        alpha = np.linspace(0, 2*np.pi, 50)
        x = np.array(np.cos(alpha))*radius_circle
        y = np.array(np.sin(alpha))*radius_circle
        yz = np.zeros((len(x),2))
        yz[:,0] = x
        yz[:,1] = y
        yz = np.delete(yz,-1,axis=0).round(2) ## delete last coordinate
        self.yz = yz
        self.AddCurrentSection()
        return yz
    
    def circle_cable_ForCableBridge(self, coord, radius = 0.05):
        alpha = np.linspace(0, 2*np.pi, 5)
        x = float(coord[0]) 
        y = float(coord[1]) + np.array(np.cos(alpha))*radius
        z = float(coord[2]) + np.array(np.sin(alpha))*radius
        xyz = np.zeros((len(y), 3))
        xyz[:, 0] = x + y*0
        xyz[:, 1] = y
        xyz[:, 2] = z
        #print(xyz)
        # self.yz = np.delete(xyz,-1,axis=0).round(2) ## delete last coordinate
        #self.AddCurrentSection()
        return xyz


    

    def A_shape_column(self, w_2, h_1, h_2, h_3, h_4, h_5, k, bridge_thick):
        """
        见示意图

        """
        w_1 = (h_1+h_2+h_3)/k
        yz1 = np.array([
            [0, (h_1+h_2+h_3+h_4+h_5)],
            [0, (h_1+h_2+h_3)],
            [h_3/k, (h_1+h_2)],
            [0, (h_1+h_2)],
            [0, h_1],
            [(h_2+h_3)/k, h_1],            
            [w_1, 0],
            [w_1+w_2, 0],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4)],
            [(w_1+w_2)-(h_1+h_2+h_3+h_4)/k, (h_1+h_2+h_3+h_4+h_5)]
        ])

        yz2 = np.zeros(yz1.shape)
        yz2[:, 0] = -yz1[:, 0]
        yz2[:, 1] = yz1[:, 1]
        yz = np.concatenate((yz1,yz2), 0)
        bridge_max_width = (h_3-bridge_thick)/k * 2
        self.yz = yz
        self.AddCurrentSection()
        return yz, bridge_max_width, h_5 



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
        """
        input: C: The accumulative cross-section got from an CrossSection instance
               trans: transformation of deck
            
        output: v: vertices
                f: faces
    
        args: n: number of cross-section
              nps: number of vertices in one cross-section
    

        """
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


class cable_stayed_bridge_build:
    def __init__(self):
        self.Width_of_Bridge = 10 # Width of the deck                   
        self.Length_of_Bridge = 500 # Length[meter]   
        self.Thickness_of_Deck = 1                     
        self.Thickness_of_Column = 5      
        self.Span_of_Column = 100

        self.height = np.array([fz(i) for i in range(self.Length_of_Bridge)])
        theta = np.array([np.arctan(float(fyd(i))) for i in range(1,self.Length_of_Bridge)])
        self.theta = np.concatenate((np.array([0]),theta),axis=0) ## only have one dimension: ndarray[0, theta1, theta2, ...] 
    

    def duplicateDistance(self):
        '''
        Args:
            w - Span between two columns
            l - Length of the Bridge
            n - number of column
        Return:
            Coordinates for each column on x - axis
        '''
        w = self.Span_of_Column
        l = self.Length_of_Bridge
        self.n = int(l/w) ## round to smaller intiger
        D = np.zeros(self.n)
        for i in range(self.n):
            D[i] = w/2 + w*i
        return D


    def column(self, w_2, h_1, h_2, h_3, h_4, h_5, k):
        
        self.h_deck_bottom = h_1 + h_2
        self.h_cable_bottom = h_1 + h_2 + h_3 + h_4
        
        cs = CrossSection()
        for i in range(self.Thickness_of_Column):
            _, self.bridge_max_width, self.h_cable = cs.A_shape_column(w_2, h_1, h_2, h_3, h_4, h_5, k, self.Thickness_of_Deck)
        
        m = Member(cs.C,None,None)
        vertices = m.v
        edges = [ ]
        faces = m.f

        npt_half = int(m.npts/2)
        f1 = ()
        f2 = ()
        for i in range(npt_half):
            f1 += (i,)
            f2 += (m.npts * (self.Thickness_of_Column - 1) + i,)
        faces.append(f1)
        faces.append(f2)

        f3 = ()
        f4 = ()
        for i in range(npt_half):
            f3 += (i + npt_half, )
            f4 += (m.npts * (self.Thickness_of_Column - 1) + i + npt_half,)
        faces.append(f3)
        faces.append(f4)

        self.D = self.duplicateDistance()
        for i in range(self.n):
            new_mesh=bpy.data.meshes.new("new_mesh")
            new_mesh.from_pydata(vertices,edges,faces)
            new_mesh.update()

            #create an object from the mesh
            column=bpy.data.objects.new("column" + str(i+1),new_mesh)
            column.location.x = self.D[i]

            #add the object to view
            view_layer=bpy.context.view_layer
            view_layer.active_layer_collection.collection.objects.link(column)


    def deck(self):
        cs = CrossSection()
        h = self.h_deck_bottom + self.Thickness_of_Deck
        for i in range(self.Length_of_Bridge):
            cs.Rectangular_deck(self.bridge_max_width, h, self.Thickness_of_Deck)

        t = np.zeros((self.Length_of_Bridge,3))
        t[:,0] = np.array([fx(i) for i in range(self.Length_of_Bridge)])
        t[:,1] = np.array([fy(i) for i in range(self.Length_of_Bridge)])
        rotvec = np.zeros((self.Length_of_Bridge,3))
        rotvec[:,2] = self.theta
        Rot = R.from_rotvec(rotvec)
        m = Member(cs.C,t,Rot.as_quat())
        vertices = m.v
        edges = [ ]
        faces = m.f
        
        f1 = ()
        f2 = ()
        for i in range(m.npts):
            f1 += (i,)
            f2 += (m.npts*self.Length_of_Bridge-i-1,)
        faces.append(f1)
        faces.append(f2)

        new_mesh=bpy.data.meshes.new("new_mesh")
        new_mesh.from_pydata(vertices,edges,faces)
        new_mesh.update()

        #create an object from the mesh
        deck=bpy.data.objects.new("deck" + str(i+1),new_mesh)

        #add the object to view
        view_layer=bpy.context.view_layer
        view_layer.active_layer_collection.collection.objects.link(deck)

    
    def cable(self, num_cable):
        """
        input: num_cable - total number of cable for one side of column

        """
        # bottom axis of cable
        z_cable_bottom = self.h_deck_bottom + self.Thickness_of_Deck/2
        
        y_cable_bottom = self.bridge_max_width/2 * (9/10)
        
        x_dist = (self.Span_of_Column/2 * 0.8)/num_cable ## distance between two cable in x direction 
        D = self.duplicateDistance()
        x_cable_bottom = np.zeros([self.n, num_cable*2]) ## one row represents x coordinate of all cable for one column
        loc = np.hstack((np.linspace(-num_cable, -1, num_cable), np.linspace(1, num_cable, num_cable)))  ## help to locate x coordinate
        for i in range(self.n):
            x_cable_bottom[i, :] = D[i] + x_dist * loc
        
        x_cable_bottom = x_cable_bottom.reshape([-1, 1])

        cable_bottom_right = np.hstack((x_cable_bottom, x_cable_bottom*0+y_cable_bottom, x_cable_bottom*0+z_cable_bottom))
        cable_bottom_left = cable_bottom_right * np.array([[1,-1,1]])
        cable_bottom = np.concatenate([cable_bottom_left, cable_bottom_right], 0) ## all the coordinate for cable bottom

        # top axis of cable
        y_cable_top = 0

        x_cable_top = np.ones([self.n, num_cable*2]) * D.reshape([self.n, 1]) # *2: front and back
        x_cable_top = x_cable_top.reshape([-1, 1])
        for i in range(self.n):
            index_even = i*2
            index_odd = i*2 + 1
            x_cable_top[(index_odd*num_cable) : ((index_odd+1)*num_cable)] += (self.Thickness_of_Column - 1)
        

        z_dist = self.h_cable/(num_cable+1)/2
        z_cable_top = np.zeros([self.n, num_cable*2]) 
        loc = np.hstack((np.linspace(num_cable, 1, num_cable), np.linspace(1, num_cable, num_cable)))
        for i in range(self.n):
            z_cable_top[i, :] = self.h_cable_bottom + self.h_cable/2 + z_dist * loc
        z_cable_top = z_cable_top.reshape([-1, 1])

        cable_top_right = np.hstack((x_cable_top, z_cable_top*0 + y_cable_top, z_cable_top))
        cable_top_left = cable_top_right * np.array([[1,-1,1]])
        cable_top = np.concatenate([cable_top_left, cable_top_right], 0) ## all the coordinate for cable top

        for i in range(self.n * num_cable * 4):
            self.v = []
            self.f = []
            cs = CrossSection()

            a = cs.circle_cable_ForCableBridge(cable_bottom[i, :])
            b = cs.circle_cable_ForCableBridge(cable_top[i, :])

            for i1 in range(a.shape[0]):
                self.v.append(a[i1, :])

            for i2 in range(b.shape[0]):
                self.v.append(b[i2, :])

            self.npts = a.shape[0]

            for j in range(self.npts):
                self.f = self.f + [(j, np.mod(j+1,self.npts),np.mod(j+1,self.npts) + self.npts, j + self.npts) for k in range(self.npts)]

            vertices = self.v
            edges = [ ]
            faces = self.f

            print(vertices)
            print(faces)

            new_mesh=bpy.data.meshes.new("new_mesh")
            new_mesh.from_pydata(vertices,edges,faces)
            new_mesh.update()

            #create an object from the mesh
            cable=bpy.data.objects.new("cable" + str(i+1),new_mesh)

            #add the object to view
            view_layer=bpy.context.view_layer
            view_layer.active_layer_collection.collection.objects.link(cable)
       

        

bridge = cable_stayed_bridge_build()
bridge.column(1,5,2,15,2,8,4)
bridge.deck()
bridge.cable(8)


