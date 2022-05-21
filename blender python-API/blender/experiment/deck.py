import bpy
import math
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


#bpy.ops.object.light_add()
#light=bpy.context.active_object #the chosen object
#light.location=[4,4,4]

#bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-8, 0, 4.05), rotation=(1.05, 0, -1.6), scale=(1, 1, 1))
#camera=bpy.context.active_object


#bpy.ops.mesh.primitive_cube_add()
#cube=bpy.context.active_object #the chosen object
#cube.location=[0,0,0]

######################create mesh

class CrossSection:

    def __init__(self): 
        self.yz=None
        self.C=None
        
    def W(self,b,d,tw,tf): 
        yz = np.array([
                [b/2,-d/2],
                [b/2,-d/2+tf],
                [tw/2,-d/2+tf],
                [tw/2,d/2-tf],
                [b/2,d/2-tf],
                [b/2,d/2],
                [-b/2,d/2],
                [-b/2,d/2-tf],
                [-tw/2,d/2-tf],
                [-tw/2,-d/2+tf],
                [-b/2,-d/2+tf],
                [-b/2,-d/2]

            ])

        self.yz = yz
        self.addsection()
        return self.yz

    def L(self,b,d,tf,tw):
        yz=np.array([
            [0,0],
            [b,0],
            [b,tf],
            [tw,tf],
            [tw,d],
            [0,d]
        ])
        self.yz=yz
        self.addsection()
        return self.yz
    
    def R(self,l,w):
        yz=np.array([
            [l/2,-w/2],
            [l/2,w/2],
            [-l/2,w/2],
            [-l/2,-w/2]
        ])
        self.yz=yz
        self.addsection()
        return self.yz
    
    def Pipe(self,R,r,inner):
        theta=np.arange(0,2*np.pi,0.01)
        yz=np.zeros((np.size(theta,0),2))
        if inner==True:
            a=np.zeros((np.size(theta,0),2))
            b=np.zeros((np.size(theta,0),2))
            a[:,0]=R*np.cos(theta)
            a[:,1]=R*np.sin(theta)
            b[:,0]=r*np.cos(theta)
            b[:,1]=r*np.sin(theta)
            yz=np.concatenate((a,b),0)
        else:
            yz[:,0]=R*np.cos(theta)
            yz[:,1]=R*np.sin(theta)
        self.yz=yz
        self.addsection()
        return self.yz
    
    def addsection(self):
        yz = self.yz.reshape((1,self.yz.shape[0],self.yz.shape[1])) #shape: 0-row 1-column #reshape: change to yz plane 1代表1面
        if self.C is None: #注意写法
            self.C = yz
        else:
            self.C = np.concatenate((self.C,yz),0) #concatenate: 矩阵叠加. 这里是在x轴上重复 (0-x 1-y 2-z) 代表一个面叠加为一个矩阵
        return self.C #所有面都重叠在一起
    
    def Show(self,double):
        if double == False:
            idx = np.mod(np.arange(self.yz.shape[0]+1),self.yz.shape[0]) #连接封闭二维图形的惯用写法
            plt.figure()  #begin drawing
            plt.plot(self.yz[idx,0],self.yz[idx,1]) #连接封闭二维图形的惯用写法
            plt.xlabel('y'); plt.ylabel('z')
            plt.axis('equal')
            plt.title("Cross-section {:d} points".format(self.yz.shape[0]))
        else:
            idx=np.mod(np.arange(int(self.yz.shape[0]/2+1)),int(self.yz.shape[0]/2))
            plt.plot(self.yz[idx,0],self.yz[idx,1])
            plt.plot(self.yz[idx+int(self.yz.shape[0]/2),0],self.yz[idx+int(self.yz.shape[0]/2),1]) #矩阵可以直接加常数
            plt.xlabel('y'); plt.ylabel('z')
            plt.axis('equal')
            plt.title("Cross-section {:d} points".format(self.yz.shape[0]))



##########################################################################################################################################
class Member:
    def __init__(self,C,t1,t2):
        self.n = C.shape[0] #number of faces
        self.npts = C.shape[1] #number of vertices in one face
        
        #################trans
        t = np.zeros((self.n,3)) #np.array(a,b): a行b列
        t[:,0] = np.arange(self.n) #第一列: 0--(self.n-1) #只要在x轴上转制，先埋好伏笔，且推进为1
        #t[:,1] = np.arange(self.n)/3
        n=self.n
        omega = np.pi / (n-1)
        t[:,1] = np.sin(omega*np.arange(n))*t1  # y direction
        t[:,2] = np.sin(omega*np.arange(n))*t2  # z direction
        self.t= t

        #################rotate
        rotvec = np.zeros((n,3))
        rotvec[:,2]=np.arctan(omega * t1 * np.cos(omega*np.arange(n))) #y direction (z rotate)
        rotvec[:,1]=-np.arctan(omega * t2 * np.cos(omega*np.arange(n)))
        self.R = R.from_rotvec(rotvec)   #把四元数转化为旋转矩阵

        self.v = [] #vertices
        self.f = [] #faces

        xyz = np.zeros((self.npts,3)) #对之前C里的每个面进行处理, xyz只代表一个面
        for i in range(self.n):
            xyz[:,1:] = C[i,:,:] #xyz所有行,第二列开始到最后一列 x坐标全是0 #这里对C的处理是把每一个面提取出来 
            c = self.R[i].apply(xyz) + self.t[i,:] #self.R[i]: 第i个旋转矩阵 #先旋转，再移动  #self.t这里的写法是broadcast
            self.v = self.v + [(c[k,0],c[k,1],c[k,2]) for k in range(self.npts)]  #(c[k,0],c[k,1],c[k,2]) is a list. 
            if i>0:
                m = self.npts * (i-1)
                idx1 = np.arange(m,m+self.npts)    #arrange包括开始 不包括结束
                idx2 = np.arange(m+self.npts,m+2*self.npts) 
                self.f = self.f + [(idx1[k],idx1[np.mod(k+1,self.npts)],idx2[np.mod(k+1,self.npts)],idx2[k]) for k in range(self.npts)]


        
###############################################################################################################################################

if __name__ == "__main__":
    cs = CrossSection()
    for i in range(20):
    # create constant rectangular cross section
        cs.R(10,5)
    cs.Show(False)
    m=Member(cs.C,2,2)
    
vertices = m.v
edges=[]
faces=m.f



new_mesh=bpy.data.meshes.new("new_mesh")
new_mesh.from_pydata(vertices,edges,faces)
new_mesh.update()


#create an object from the mesh
new_object=bpy.data.objects.new("new_object",new_mesh)

#add the object to view
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(new_object)

#mod_skin=new_object.modifiers.new('Skin','SKIN')