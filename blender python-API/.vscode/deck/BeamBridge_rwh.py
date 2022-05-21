import bpy
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

###############################################

############# Import of Geometry ##############

###############################################

## a_beams, b_beams etc. ##
# Unit: meter

ab=6
bb=2
cb=2
db=6
eb=2
fb=2
gb=2
hb=2
ib=1
jb=cb/2
kb=ab+bb
Length_of_Bridge=1000
Span_of_Columns=20
Width_of_Columns=2 ; Width_of_Columns+=1
Width_of_Beam=0.7
Height_of_Beams=1

###############################################

################ CrossSections ################

###############################################


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


    def Column2(self,a,b,c,d,e,f):
        yz=np.array([
            [c,d],
            [a,0],
            [a+b,0],
            [c+b,d],
            [c+b,d+e+f],
            [c,d+e+f],
            [c,d+e],
            [-c,d+e],
            [-c,d+e+f],
            [-(c+b),d+e+f],
            [-(c+b),d],
            [-(a+b),0],
            [-a,0],
            [-c,d]
        ])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def Deck2(self,b,c,k,i,j,h):
        x=Width_of_Beam
        y=Height_of_Beams
        yz=np.array([
            [j,h],
            [c,0],
            [c+b,0],
            [b+c+k/3-x,h/3-x*h/k],
            [b+c+k/3-x,h/3-x*h/k-y],
            [b+c+k/3,h/3-x*h/k-y],
            [b+c+k/3,h/3],
            [b+c+k*2/3-x,h*2/3-x*h/k],
            [b+c+k*2/3-x,h*2/3-x*h/k-y],
            [b+c+k*2/3,h*2/3-x*h/k-y],
            [b+c+k*2/3,h*2/3],
            [b+c+k-x,h-x*h/k],
            [b+c+k-x,h-x*h/k-y],
            [b+c+k,h-x*h/k-y],
            [c+b+k,h],
            [c+b+k,h+i],
            [-(c+b+k),h+i],
            [-(c+b+k),h],
            [-b-c-k,h-h*x/k-y],
            [-b-c-k+x,h-h*x/k-y],
            [-b-c-k+x,h-h*x/k],
            [-b-c-2*k/3,2*h/3],
            [-b-c-2*k/3,2*h/3-h*x/k-y],
            [-b-c-2*k/3+x,2*h/3-h*x/k-y],
            [-b-c-2*k/3+x,2*h/3-h*x/k],
            [-b-c-k/3,h/3],
            [-b-c-k/3,h/3-h*x/k-y],
            [-b-c-k/3+x,h/3-h*x/k-y],
            [-b-c-k/3+x,h/3-h*x/k],
            [-(c+b),0],
            [-c,0],
            [-j,h]
        ])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def Beam_right(self,b,c,k,g,f,e,d,y,h):
        yz=np.array([
            [b+c,g+f+e+d],
            [b+c,g+f+e+d-y],
            [b+c+k,g+f+e+d-y+h],
            [b+c+k,g+f+e+d+h]
        ])
        self.yz = yz
        self.AddCurrentSection()
        return yz

    def Beam_left(self,b,c,k,g,f,e,d,y,h):
        yz=np.array([
            [-(b+c),g+f+e+d],
            [-(b+c),g+f+e+d-y],
            [-(b+c+k),g+f+e+d-y+h],
            [-(b+c+k),g+f+e+d+h]
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
    n = Length_of_Bridge+Width_of_Columns
    # create constant rectangular cross section
    for i in range(n):
        #cs.Rectangule(50.0, 5.0)
        cs.Deck2(bb,cb,kb,ib,jb,hb)
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
deck=bpy.data.objects.new("deck",new_mesh)
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(deck)

deck.location[2] =gb+fb+eb+db  # [0,1,2] for x,y,z respectively

###############################################

################### Girders ###################

###############################################

if __name__ == "__main__":
    cs = CrossSection()
    n = Length_of_Bridge+Width_of_Columns
    # create constant rectangular cross section
    for i in range(n):
        #cs.Rectangule(50.0, 5.0)
        cs.Rectangule(bb,gb)
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
beam=bpy.data.objects.new("beam",new_mesh)

#add the object to view
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(beam)

beam2=bpy.data.objects.new('beam2', beam.data)
bpy.data.collections["Collection"].objects.link(beam2)

beam.location[2]=fb+eb+db+gb/2
beam.location[1]=cb+bb/2
beam2.location[2]=fb+eb+db+gb/2
beam2.location[1]=-(cb+bb/2)



###############################################

############### Horizontal Beam ###############

###############################################


if __name__ == "__main__":
    n=Width_of_Columns
    cs = CrossSection()
    y=Height_of_Beams*1.2
    for i in range(n):
        cs.Beam_right(bb,cb,kb,gb,fb,eb,db,y,hb)
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
beam_right=bpy.data.objects.new("beam_right",new_mesh)
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(beam_right)



if __name__ == "__main__":
    n=Width_of_Columns
    cs = CrossSection()
    y=Height_of_Beams*1.2
    for i in range(n):
        cs.Beam_left(bb,cb,kb,gb,fb,eb,db,y,hb)
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
Beam_left=bpy.data.objects.new("Beam_left",new_mesh)
view_layer=bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(Beam_left)

###############################################

################### Columns ###################

###############################################


if __name__ == "__main__":
    cs = CrossSection()
    n = Width_of_Columns
    # create constant rectangular cross section
    for i in range(n):
        #cs.Rectangule(50.0, 5.0)
        cs.Column2(ab,bb,cb,db,eb,fb)
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

# n number_of_columns

# l total_length

# Distance away from the y-axis
def duplicate_distance(w,l):
    n=int(l/w)
    D=np.zeros(n+1)
    for i in range(n+1):
        D[i]=l/n*i
    return D


for i,j in zip(duplicate_distance(Span_of_Columns,Length_of_Bridge),range(len(duplicate_distance(Span_of_Columns,Length_of_Bridge)))):
    if j == 0:
        continue
    else:
        column = bpy.data.objects["column"]
        columnzip=bpy.data.objects.new('column%1.f' %j, column.data)
        columnzip.location.x = i
        bpy.data.collections["Collection"].objects.link(columnzip)
        
        beam_right = bpy.data.objects["beam_right"]
        beam_rightzip=bpy.data.objects.new('beam_right%1.f' %j, beam_right.data)
        beam_rightzip.location.x = i
        bpy.data.collections["Collection"].objects.link(beam_rightzip)

        Beam_left = bpy.data.objects["Beam_left"]
        beam_leftzip=bpy.data.objects.new('Beam_left%1.f' %j, Beam_left.data)
        beam_leftzip.location.x = i
        bpy.data.collections["Collection"].objects.link(beam_leftzip)
    
