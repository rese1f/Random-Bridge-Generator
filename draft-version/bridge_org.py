import bpy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *
import math

cp = bpy.data.texts["component.py"].as_module() 
ut = bpy.data.texts["utils.py"].as_module()

class BeamBridge:
    def __init__(self, l_deck = 200, h_deck = 11, t_deck = 1, w_deck = 10, \
        d_column = 10, l_column = 2, w_column = 4, h_column = 10):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        # self.column_height = h_column
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column

        self.column_thickness = 1
        self.deck_index = 1 # random.choice([1,2,3,4,5,6])
        self.column_index = random.choice([0,1])
        self.column_number = random.choice([1,2])

        self.bearing_length = 1
        self.bearing_thickness = random.choice([0,0.15,0.2])
        self.bearing_width = 0.5

        self.piercap_thickness = 1
        self.piercap_index = random.choice([0,1])

        self.girder_width = random.choice([0.6,0]) 
        self.girder_thickness = self.girder_width
        self.girder_index = random.choice([1,2])

        self.parapet_width = 0.2
        self.parapet_thickness = 1

        self.column_height = self.deck_height-self.deck_thickness\
            -self.bearing_thickness-self.piercap_thickness-self.girder_thickness

        self.railslab_width = 2
        self.railslab_thickness = 0.2

        self.rail_width = 0.1
        self.rail_thickness = 0.2

        self.sleeper_height = 0.2
        self.sleeper_length = 0.2
        self.sleeper_span = 0.5


        self.column_number = int(self.deck_length/self.column_distance)      
        self.sleeper_number = int(self.deck_length/self.sleeper_span)

        self.a = (np.random.rand())*20
        self.b = (np.random.rand())*1000

        self.theta = np.array([np.arctan(float(self.fyd(i,self.a,self.b))) for i in range(1,self.deck_length+1)])        

        self.deck()
        self.column()
        self.girder_lateral()
        self.girder_longitude()
        self.girder_bearing()
        self.parapet()
        self.rail()

#############################################################################################
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

###############################################################################################

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
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'concrete solid deck', t, quat, self.deck_width, None)
            member.concrete_solid()
        
        elif self.deck_index == 2:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'concrete PK deck', t, quat, self.deck_width, None)
            member.concrete_PK()
        
        elif self.deck_index == 3:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'concrete box deck', t, quat, self.deck_width, None)
            member.concrete_box()
        
        elif self.deck_index == 4:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'concrete constalia deck', t, quat, self.deck_width, None)
            member.concrete_costalia()

        elif self.deck_index == 5:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel box deck', t, quat, self.deck_width, None)
            member.steel_box() 

        elif self.deck_index == 6:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel sidebox deck', t, quat, self.deck_width, None)
            member.steel_sidebox()

    
    def column(self):

        if self.column_number == 1:
            member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'column')
            # (self, w_column, h_column, t_column, l_column, h_deck_t, t_deck, name)
    
            if self.column_index == 0:
                member.rectangle_column()

                for i in range(self.column_number):
                    theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
                    ut.Duplicate(member.column, 'rectangle_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(theta))

            elif self.column_index == 1:
                member.cylinder_column()

                for i in range(self.column_number):
                    theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
                    ut.Duplicate(member.column, 'cylinder_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, -math.pi/2, np.arctan(theta))
            
            member.column.select_set(True)
            bpy.ops.object.delete()
        
        else:
            self.column_width = 1.5
            member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'column')            
            dis = 2.5
            if self.column_index == 0:
                member.rectangle_column()
                for i in range(self.column_number):
                    theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
                    ut.Duplicate(member.column, 'rectangle_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                        self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta), None, 0, 0, np.arctan(theta))
                    ut.Duplicate(member.column, 'rectangle_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                        self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta), None, 0, 0, np.arctan(theta))
            
            elif self.column_index == 1:
                member.cylinder_column()

                for i in range(self.column_number):
                    theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
                    ut.Duplicate(member.column, 'cylinder_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                        self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta), None, 0, -math.pi/2, np.arctan(theta))            
                    ut.Duplicate(member.column, 'rectangle_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                        self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta), None, 0, -math.pi/2, np.arctan(theta))
        
            member.column.select_set(True)
            bpy.ops.object.delete()            
        # Duplicate(member.column, 'jack', 10, 10, 10, None, math.pi/2, None)    


    def girder_lateral(self):
        member = cp.Girder(self.girder_width, self.girder_thickness, self.deck_length, self.deck_height, self.deck_thickness, self.deck_width,\
            self.column_length, self.bearing_thickness, self.bearing_width, 'lateral girder') 
        # w_girder=0.6, t_girder=0.6, l_deck=10, h_deck=10, t_deck=1, w_deck=10, l_column=1.5, t_bearing=0.2, w_bearing=0.6, name='girder'
        
        if self.piercap_index == 0:
            member.rectangle_pier_cap()

        elif self.piercap_index == 1:
            member.box_pier_cap()
        
        for i in range(self.column_number):            
            ut.Duplicate(member.girder_lateral, 'lateral girder'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(float(self.fyd(int(i*self.column_distance),self.a,self.b))))                  
        
        member.girder_lateral.select_set(True)
        bpy.ops.object.delete()
    

    def girder_longitude(self):
        # (-1)**i*self.W_deck/5*int((i+1)/2)
        for j in range(5):
            dis = (-1)**j*self.deck_width/5*int((j+1)/2)
            n = self.deck_length
            t = np.zeros((n,3))
            t[:,0] = np.array([(self.fx(i,self.a,self.b) - dis*np.sin(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            t[:,1] = np.array([(self.fy(i,self.a,self.b) + dis*np.cos(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            rotvec = np.zeros((n,3))
            rotvec[:,2] = self.theta
            # rotvec[:,2] = np.arraynp.arctan([float(self.fyd(int(i),self.a,self.b)) for i in range(n)])
            Rot = R.from_rotvec(rotvec)
            quat = Rot.as_quat()

            member = cp.Girder(self.girder_width, self.girder_thickness, self.deck_length, self.deck_height, self.deck_thickness, self.deck_width,\
                self.column_length, self.bearing_thickness, self.bearing_width, 'longitude girder' + str(j+1), t, quat)
            
            if self.girder_index == 1:
                member.I_girder()
            else:
                member.rectangle_girder()
    

    def girder_bearing(self):
        name = 'gird bearing'
        member = cp.GirderBearing(self.deck_width, self.deck_height, self.deck_thickness, self.bearing_thickness,\
             self.bearing_length, self.girder_thickness, name)
        #w_deck=10, h_deck=8, t_deck=1, t_bearing=0.5, l_bearing=1, t_girder=1, name='girder_bearing'
        
        for i in range(self.column_number):
            theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
            ut.Duplicate(member.girder_bearing, 'gird bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(theta))                    
        
        member.girder_bearing.select_set(True)
        bpy.ops.object.delete()
    
    def parapet(self):
        #self, w_deck, l_deck, h_deck, t_deck, w_parapet, t_parapet,name
        for j in range(2):
            dis = (-1) ** j * (self.deck_width*0.5 - self.parapet_width/2)
            n = self.deck_length
            t = np.zeros((n,3))
            t[:,0] = np.array([(self.fx(i,self.a,self.b) - dis*np.sin(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            t[:,1] = np.array([(self.fy(i,self.a,self.b) + dis*np.cos(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            rotvec = np.zeros((n,3))
            rotvec[:,2] = self.theta
            # rotvec[:,2] = np.arraynp.arctan([float(self.fyd(int(i),self.a,self.b)) for i in range(n)])
            Rot = R.from_rotvec(rotvec)
            quat = Rot.as_quat()
            member = cp.Parapet(self.deck_width, self.deck_length, self.deck_height, self.deck_thickness, self.parapet_width, self.parapet_thickness, 'parapet', t, quat) 
            member.rectangleParapet()

    def rail(self):
        # w_deck, l_deck, h_deck, t_deck, w_railslab, t_railslab,\
        # w_rail, t_rail, h_sleeper, l_sleeper, span_sleeper, name, tran, quat

        # slab
        for j in range(2):
            dis = (-1) ** j * (self.deck_width*0.3 - self.railslab_width/2)
            n = self.deck_length
            t = np.zeros((n,3))
            t[:,0] = np.array([(self.fx(i,self.a,self.b) - dis*np.sin(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            t[:,1] = np.array([(self.fy(i,self.a,self.b) + dis*np.cos(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            rotvec = np.zeros((n,3))
            rotvec[:,2] = self.theta
            # rotvec[:,2] = np.arraynp.arctan([float(self.fyd(int(i),self.a,self.b)) for i in range(n)])
            Rot = R.from_rotvec(rotvec)
            quat = Rot.as_quat()
            member = cp.Railway(self.deck_width, self.deck_length, self.deck_height, self.deck_thickness,\
                 self.railslab_width, self.railslab_thickness, self.rail_width, self.rail_thickness,\
                    self.sleeper_height, self.sleeper_length,\
                        self.sleeper_span, 'rail slab', t, quat) 
            member.railSlabBuild() 



        # sleeper
        member = cp.Railway(self.deck_width, self.deck_length, self.deck_height, self.deck_thickness,\
            self.railslab_width, self.railslab_thickness, self.rail_width, self.rail_thickness,\
            self.sleeper_height, self.sleeper_length,\
            self.sleeper_span, 'rail sleeper', None, None)

        member.railSleeperBuild()     

        for i in range(self.sleeper_number):
            theta = float(self.fyd(int(i*self.sleeper_span),self.a,self.b))
            for j in range(2):
                dis = (-1) ** j * (self.deck_width*0.3 - self.railslab_width/2)

                ut.Duplicate(member.rail_sleeper, 'rail sleeper'+str(i+1), self.fx(int(i*self.sleeper_span),self.a,self.b) - dis*np.sin(theta),\
                    self.fy(int(i*self.sleeper_span),self.a,self.b) + dis*np.cos(theta), None, 0, 0, np.arctan(theta))
        

        member.rail_sleeper.select_set(True)
        bpy.ops.object.delete()        


        # rail track
        for j in range(2):
            dis = (-1) ** j * (self.deck_width*0.3 - self.railslab_width/5)       
            n = self.deck_length
            t = np.zeros((n,3))
            t[:,0] = np.array([(self.fx(i,self.a,self.b) - dis*np.sin(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            t[:,1] = np.array([(self.fy(i,self.a,self.b) + dis*np.cos(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            rotvec = np.zeros((n,3))
            rotvec[:,2] = self.theta
            # rotvec[:,2] = np.arraynp.arctan([float(self.fyd(int(i),self.a,self.b)) for i in range(n)])
            Rot = R.from_rotvec(rotvec)
            quat = Rot.as_quat()
            member = cp.Railway(self.deck_width, self.deck_length, self.deck_height, self.deck_thickness,\
                 self.railslab_width, self.railslab_thickness, self.rail_width, self.rail_thickness,\
                    self.sleeper_height, self.sleeper_length,\
                        self.sleeper_span, 'rail track', t, quat) 
            member.railBuild()
        
        for j in range(2):
            dis = (-1) ** j * (self.deck_width*0.3 - self.railslab_width*4/5)       
            n = self.deck_length
            t = np.zeros((n,3))
            t[:,0] = np.array([(self.fx(i,self.a,self.b) - dis*np.sin(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            t[:,1] = np.array([(self.fy(i,self.a,self.b) + dis*np.cos(float(self.fyd(i,self.a,self.b)))) for i in range(n)])
            rotvec = np.zeros((n,3))
            rotvec[:,2] = self.theta
            # rotvec[:,2] = np.arraynp.arctan([float(self.fyd(int(i),self.a,self.b)) for i in range(n)])
            Rot = R.from_rotvec(rotvec)
            quat = Rot.as_quat()
            member = cp.Railway(self.deck_width, self.deck_length, self.deck_height, self.deck_thickness,\
                 self.railslab_width, self.railslab_thickness, self.rail_width, self.rail_thickness,\
                    self.sleeper_height, self.sleeper_length,\
                        self.sleeper_span, 'rail track', t, quat) 
            member.railBuild()        
    
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
        member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'deck')
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
    
        member = cp.Archring(self.archring_thickness, self.archring_height, self.archring_length, self.archring_width, 'arch', t, quat)
        
        
        
    def uprightcolumn(self):
        n = self.column_number
        for i in range (n):
            self.column_height = 1
            
        member = cp.Uprightcolmn(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'Uprightcolumn')
        
        
        for i in range (n):
            
            ut.Duplicate(member.uprightcolumn, 'uprightcolumn'+str(i+1), self.fx(int(i*self.column_span)), self.fy(int(i*self.column_span)) + 1*self.deck_width/4,\
                        self.fz(int(i*self.column_span)), 0, 0, 0, None, None, self.deck_height-self.deck_thickness-self.bearing_thickness-self.fz(i*self.column_span))
       
            ut.Duplicate(member.uprightcolumn, 'uprightcolumn'+str(i+1), self.fx(int(i*self.column_span)), self.fy(int(i*self.column_span)) - 1*self.deck_width/4,\
                        self.fz(int(i*self.column_span)), 0, 0, 0, None, None, self.deck_height-self.deck_thickness-self.bearing_thickness-self.fz(i*self.column_span))
       
        
        member.uprightcolumn.select_set(True)
        bpy.ops.object.delete()


    def bearing(self):
        a = self.column_length
        T = self.bearing_thickness
        t = T/3
        d = a

        member = cp.ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
        
            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_span)),\
                self.fy(int(i*self.column_span)) + 1*self.deck_width/4, self.deck_height - self.bearing_thickness/2 - self.deck_thickness,0,math.pi/2,0, None, None, 0.5)

            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_span)),\
                self.fy(int(i*self.column_span)) - 1*self.deck_width/4, self.deck_height - self.bearing_thickness/2 - self.deck_thickness,0,math.pi/2,0, None, None, 0.5)
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()
       
        
    def pier(self):
        
        member = cp.Pier(self.pier_width, self.pier_height, self.pier_thickness, "pier")

        ut.Duplicate(member.pier, 'pier', self.deck_length - self.pier_thickness/3, None, None,0, 0, 0, None, None, 1)

class CableStayedBridge:
    def __init__(self, l_deck = None, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0, l_column = 2, w_column = 10, h_column = 36):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_height = h_column
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column


        # deck_length
        # deck_height
        # deck_thickness
        # deck_width
        # column_distance
        # column_length (x)
        # column_width (yz plane)
        # column_height
        
        self.column_thickness = 1
        self.bearing_thickness = 0.3
        self.cable_function = None
        self.cable_top = None
        self.cable_bottom = None
        self.truss_thickness = None
        self.cable_number = 7

    ########################################
        num_column_list = [1, 2]
        self.column_number = random.choice(num_column_list)

        index_column_list = [1,2,3,4]
        self.column_index = random.choice(index_column_list)

        if self.column_index == 1 or self.column_index == 2 or self.column_index == 3:
            index_deck_list = [1,2,4,5,6]
            self.cable_face = 2
        elif self.column_index == 4:
            index_deck_list = [3]
            self.cable_face = 1
        self.deck_index = random.choice(index_deck_list)

        if self.column_index == 1:
            index_cable_list = [1,2]
        else:
            index_cable_list = [1,2,3]
        self.cable_index = random.choice(index_cable_list)

        tru_list = [0,1]
        tru_list = [0]
        self.truss = random.choice(tru_list)



        if self.column_number == 1:
            a = random.uniform(1.5, 3)
            self.deck_length = a * self.column_height
        else:
            a = random.uniform(3, 6)
            self.column_distance = a * self.column_height
            self.deck_length = 2 * self.column_distance
        
        self.column()
        self.deck()
        self.cable()
        self.cablebase()
        self.cabletop()
        self.bearing()
        
    def column(self):
        self.deck_height -= self.bearing_thickness
        # self.column_index = 1
        for i in range(self.column_number):        
            if self.column_index == 1:
                member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'A1 column') # 'A1 column'
                member.A1()

            elif self.column_index == 2:
                member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'double column')
                member.double()

            elif self.column_index == 3:
                member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'door column')
                member.door()
            
            elif self.column_index == 4:
                member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length, self.deck_height, self.deck_thickness, 'tower column')
                member.tower()

            self.cable_function = member.cable_function
            member.column.location.x = (-1)**(i+1) * self.column_distance/2

        self.deck_height += self.bearing_thickness

    def deck(self):
        k = self.cable_function[2]
        b_in = self.cable_function[4]
        h = self.deck_height + self.deck_thickness
        if b_in == 0:
            self.deck_width = self.column_width - self.column_thickness * 2
        else:
            self.deck_width = - (h - b_in)/k * 2  

#        self.deck_index = 6
        
        if self.truss == 0:
            if self.deck_index == 1:
                # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete solid deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete solid deck', None, None, self.deck_width, None)
                member.concrete_solid()
            
            elif self.deck_index == 2:
                #member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete PK deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete PK deck', None, None, self.deck_width, None)
                member.concrete_PK()
            
            elif self.deck_index == 3:
                #member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete box deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete box deck', None, None, self.deck_width, None)
                member.concrete_box()
            
            elif self.deck_index == 4:
                #member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete constalia deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete constalia deck', None, None, self.deck_width, None)
                member.concrete_costalia()

            elif self.deck_index == 5:
                #member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel box deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel box deck', None, None, self.deck_width, None)
                member.steel_box() 

            elif self.deck_index == 6:
                #member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel sidebox deck', None, self.cable_function)
                member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, \
                    self.column_width, self.column_thickness, 'steel sidebox deck', \
                        None, None, self.deck_width, None)
                member.steel_sidebox()
        
        elif self.truss == 1:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'truss', None, self.cable_function) 
            member.truss()
    

    def cable(self):
        z_start = self.cable_function[0]
        z_end = self.cable_function[1]
        k = self.cable_function[2]
        b = self.cable_function[3]
        y_top = self.cable_function[5]

        
        column_loc = np.zeros(self.column_number)
        for i in range(self.column_number):
            column_loc[i] = (-1)**(i+1) * self.column_distance/2

        # top right side
        dist_top = (z_end - z_start) / (self.cable_number - 1)   
        y_cable_top0 = np.zeros([self.cable_number, 1])
        z_cable_top0 = np.zeros([self.cable_number, 1])
        z_rand = random.uniform(z_start + (z_end - z_start)/4, z_end - (z_end - z_start)/4) ## for cable index3
        for i in range(self.cable_number):
            if self.cable_index == 3:
                z_cable_top0[i] = z_rand
                dist_top = 0
            else:
                z_cable_top0[i] = z_end - i * dist_top
            
            if self.cable_face == 1:
                y_cable_top0[i] = 0
            elif self.cable_face == 2:
                if b == 0:
                    y_cable_top0[i] = y_top
                else:
                    y_cable_top0[i] = -(z_end - i * dist_top - b)/k

        
        x_cable_top = np.ones([self.column_number, self.cable_number*2]) * column_loc.reshape([self.column_number, 1]) # *2: front and back
        x_cable_top = x_cable_top.reshape([-1, 1])
        for i in range(self.column_number): ## adjust the top part of cable just in touch with the surface of column
            index_even = i*2
            index_odd = i*2 + 1
            x_cable_top[(index_odd*self.cable_number) : ((index_odd+1)*self.cable_number)] += (self.column_length/2)
            x_cable_top[(index_even*self.cable_number) : ((index_even+1)*self.cable_number)] -= (self.column_length/2) 

        yz_cable_top0 = np.concatenate((y_cable_top0, z_cable_top0), 1)
        yz_cable_top = yz_cable_top0
        for i in range(self.column_number*2 - 1):
            yz_cable_top = np.concatenate((yz_cable_top, yz_cable_top0), 0)
        
        cable_top_right = np.concatenate((x_cable_top, yz_cable_top), 1)

        # bottom right side
        if self.truss == 1:
            z_cable_bottom = self.deck_height - self.deck_thickness + self.truss_thickness
        elif self.truss ==0:    
            z_cable_bottom = self.deck_height

        if self.cable_face == 2:
            print(self.deck_width)  
            y_cable_bottom = self.deck_width/2 * (7/10)
        elif self.cable_face == 1:
            y_cable_bottom = 0

        a = random.uniform(7/8, 9/10)  
        x_L = self.deck_length / self.column_number / 2 * a ## outer distance of cable at bottom

        if self.cable_index == 2:
            k_cable = (z_end - z_cable_bottom)/x_L
            dist_bottom = dist_top / k_cable ## distance between two cable in x direction 
        else:
            b = random.uniform(1/7, 1/5)
            # x_L_in = b * x_L
            dist_bottom = x_L / self.cable_number

        x_cable_bottom = np.zeros([self.column_number, self.cable_number*2]) ## one row represents x coordinate of all cable for one column
        loc = np.hstack((np.linspace(-self.cable_number, -1, self.cable_number), np.linspace(self.cable_number, 1, self.cable_number)))  ## help to locate x coordinate
        for i in range(self.column_number):
            x_cable_bottom[i, :] = column_loc[i] + dist_bottom * loc

        x_cable_bottom = x_cable_bottom.reshape([-1, 1])
        y_cable_bottom = x_cable_bottom * 0 + y_cable_bottom
        z_cable_bottom = x_cable_bottom * 0 + z_cable_bottom
        cable_bottom_right = np.concatenate((x_cable_bottom, y_cable_bottom, z_cable_bottom), 1)

        if self.cable_face == 1:
            cable_top = cable_top_right
            cable_bottom = cable_bottom_right

            for i in range(self.column_number * self.cable_number * 2):
                cp.Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

        elif self.cable_face == 2:
            cable_top_left = cable_top_right * np.array([[1, -1, 1]])
            cable_bottom_left = cable_bottom_right * np.array([[1, -1, 1]])
            cable_top = np.concatenate((cable_top_left, cable_top_right), 0)
            cable_bottom = np.concatenate((cable_bottom_left, cable_bottom_right), 0)        

            for i in range(self.column_number * self.cable_number * 4):
                cp.Cable(cable_bottom[i, :], cable_top[i, :], "cable" + str(i+1))

        self.cable_top = cable_top
        self.cable_bottom = cable_bottom

    def cablebase(self):
        t1 = random.uniform(0.8, 1.4) # need further revise
        t2 = random.uniform(0.4, 0.7)
        t3 = random.uniform(0.8, 1.5)
        r2 = 0.15
        for i in range(self.cable_top.shape[0]):
            kxy = (self.cable_top[i, 1] - self.cable_bottom[i, 1])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
            kxz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
            kyz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 1] - self.cable_bottom[i, 1])                        
            turn = kxz/abs(kxz)

            member = cp.CableBase(t1, t2, t3, r2, 'cable_base' + str(i+1), turn)
            base = member.cable_base

            theta_z = math.atan(kxy)
            theta_y = math.atan(kxz)
            theta_x = math.atan(kyz)
            base.rotation_euler[0] = -theta_x
            base.rotation_euler[1] = -theta_y
            base.rotation_euler[2] = theta_z
            base.location.x = self.cable_bottom[i, 0]
            base.location.y = self.cable_bottom[i, 1]                        
            base.location.z = self.cable_bottom[i, 2]
    
    def cabletop(self):
        t1 = random.uniform(0.8, 1.4) # need further revise
        t2 = random.uniform(0.4, 0.7)
        t3 = random.uniform(0.8, 1.5)
        r1 = 0.08
        for i in range(self.cable_top.shape[0]):
            kxy = (self.cable_top[i, 1] - self.cable_bottom[i, 1])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
            kxz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 0] - self.cable_bottom[i, 0])
            kyz = (self.cable_top[i, 2] - self.cable_bottom[i, 2])/(self.cable_top[i, 1] - self.cable_bottom[i, 1])                        
            turn = kxz/abs(kxz)

            member = cp.CableBase(t1, t2, t3, r1, 'cable_top' + str(i+1), turn)
            top = member.cable_base

            theta_z = math.atan(kxy)
            theta_y = math.atan(kxz)
            theta_x = math.atan(kyz)
            top.rotation_euler[0] = -theta_x
            top.rotation_euler[1] = -theta_y
            top.rotation_euler[2] = theta_z
            top.location.x = self.cable_top[i, 0]
            top.location.y = self.cable_top[i, 1]                        
            top.location.z = self.cable_top[i, 2]       

    def bearing(self):
        a = self.column_length
        T = self.bearing_thickness
        t = T/3
        d = a
        for i in range(self.column_number):
            member1 = cp.DeckBearing(a, d, T, t, "bearing" + str(i*2+1))
            member1.deck_bearing.location.x = (-1)**(i+1) * self.column_distance/2
            member1.deck_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            member1.deck_bearing.location.y = a/2 + 0.5

            member2 = cp.DeckBearing(a, d, T, t, "bearing" + str(i*2+2))
            member2.deck_bearing.location.x = (-1)**(i+1) * self.column_distance/2
            member2.deck_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            member2.deck_bearing.location.y = -(a/2 + 0.5)

class SuspensionBridgeGenerator:
    def __init__(self, l_deck = 0, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0,\
                 l_column = 2, w_column = 10, h_column = 36):
        #  (self, l_deck = None, h_deck = 8, t_deck = 1, w_deck = 10, d_column = 0, l_column = 2, w_column = 10, h_column = 36):    
        
        
        t_column = 1
        index_cable = 1 
        face_cable = 2 
        num_cable = 7 
        truss = 0
        cable_span = 2

        index_column_list = [2,3]
        index_column = random.choice(index_column_list)

        if index_column == 1 or index_column ==2 or index_column == 3:
            index_deck_list = [1,2,4,5,6]
            face_cable = 2
        elif index_column == 4:
            index_deck_list = [3]
            face_cable = 1
        index_deck = random.choice(index_deck_list)    
        
        self.num_column = 2
        self.H_column = h_column
        self.T_column = t_column
        self.L_column = l_column
        self.W_column = w_column
        self.H_deck = h_deck
        self.T_deck = t_deck
        self.W_deck = w_deck
        self.index_column = index_column
        self.index_deck = index_deck
        self.index_cable = index_cable
        self.face_cable = face_cable
        self.num_cable = num_cable
        self.cable_span = cable_span
        self.tru = truss
        

        a = random.uniform(3, 6)
        self.dist_column = a * self.H_column
        self.L_deck = 2 * self.dist_column
        
        self.column()
        self.deck()
        #self.cable()
        self.mainCable()

    def fx1(self,x):
        return x-self.dist_column

    def fx2(self,x):
        return x-self.dist_column/2

    def fx3(self,x):
        return x+self.dist_column/2
    
    def fz1(self,x):
        return (0.9*self.H_column-self.H_deck)/(self.dist_column/2)**2*x**2+self.H_deck

    def fz2(self,x):
        return (0.9*self.H_column-1.2*self.H_deck)/(self.dist_column/2)**2*(x-self.dist_column/2)**2+1.2*self.H_deck

    def fz3(self,x):
        return (0.9*self.H_column-self.H_deck)/(self.dist_column/2)**2*(x-self.dist_column/2)**2+self.H_deck


    def column(self):

        for i in range(self.num_column):        
            member = cp.Column(self.W_column, self.H_column, self.T_column, \
                self.L_column, self.H_deck, self.T_deck, 'column')
            if self.index_column == 1:
                member.A1()
            elif self.index_column == 2:
                member.double()
            elif self.index_column == 3:
                member.door()
            elif self.index_column == 4:
                member.tower()

            self.cable_function = member.cable_function
            member.column.location.x = (-1)**(i+1) * self.dist_column/2


    def deck(self):
        member = cp.Slab(self.T_deck, self.H_deck, self.L_deck, self.W_column, \
            self.T_column, 'deck', None,None,self.W_deck,None)
        if self.index_deck == 1:
            member.concrete_solid()
        elif self.index_deck == 2:
            member.concrete_PK()
        elif self.index_deck == 3:
            member.concrete_box()
        elif self.index_deck == 4:
            member.concrete_costalia()
        elif self.index_deck == 5:
            member.steel_box() 
        elif self.index_deck == 6:
            member.steel_sidebox()


    def mainCable(self):

        cables = cp.Cable2d_s("cable_straight", 2, 0.05)

        for i in range(1,4):
            for j in range(1,3):
                if i == 1:
                    functionx = self.fx1
                    function = self.fz1
                    length = int(self.dist_column/2)
                elif i == 2:
                    functionx = self.fx2
                    function = self.fz2
                    length = int(self.dist_column)
                elif i == 3:
                    functionx = self.fx3
                    function = self.fz3
                    length = int(self.dist_column/2) 

                t = np.zeros((length,3))
                t[:,0] = np.array([functionx(i) for i in range(length)])
                t[:,1] = (self.W_column/2 - self.T_column/2) * (-1)**j
                t[:,2] = np.array([function(i) for i in range(length)])
                name = "main_cable" + str(i)
                cp.Cable2d(name, length, t, 0.1)

                cable_loc = ut.duplicateDistance(self.cable_span, length)
                for m,n in enumerate(cable_loc):
                    ut.Duplicate(cables.cable, "cable_straight"+str(m), functionx(n), ((self.W_column/2 - self.T_column/2) * (-1)**j),
                    self.H_deck, 0, -np.pi/2, 0, function(n) - self.H_deck, None, None)





    def cable(self):
        cable_start = np.array([0,0,0])
        cable_end = np.array([1,0,0])
        cp.Cable(cable_start, cable_end, "cable_3", cable_radius = 0.05, num = 12)
     
class SteelGirderBridge:
    def __init__(self, l_deck = 500, h_deck = 11, t_deck = 1, w_deck = 10, \
        d_column = 10, l_column = 2, w_column = 9):
        
        t_column = 1

        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column
        self.column_thichkness = t_column
        
        
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
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete solid deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'comcrete solid deck',\
            #         None, None,self.deck_width,None)
            member.concrete_solid()
        
        elif self.deck_index == 2:
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete PK deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'comcrete PK deck',\
            #         None, None,self.deck_width,None)
            member.concrete_PK()
        
        elif self.deck_index == 3:
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete box deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'comcrete box deck',\
            #         None, None,self.deck_width,None)
            member.concrete_box()
        
        elif self.deck_index == 4:
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'comcrete constalia deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'comcrete constalia deck',\
            #         None, None,self.deck_width,None)
            member.concrete_costalia()

        elif self.deck_index == 5:
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'steel box deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'steel box deck',\
            #         None, None,self.deck_width,None)
            member.steel_box() 

        elif self.deck_index == 6:
            member = cp.Slab_smy(self.deck_thickness, self.deck_height, self.deck_length, self.deck_width, 'steel sidebox deck', t, quat)
            # member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length,\
            #     self.column_width, self.column_thickness, 'steel sidebox deck',\
            #         None, None,self.deck_width,None)
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
        
        member = cp.Beam(self.beam_width, self.beam_height, self.beam_length, self.webplate_thickness, self.flange_thickness, self.deck_height, self.deck_thickness, 'i_beam', t, quat)

        for i in range(self.beam_length):
            
            theta = float(self.fyd(i,self.a,self.b))

        ut.Duplicate(member.beam, 'beam1', None, 3*self.deck_width/8, None, 0,0, None)

        ut.Duplicate(member.beam, 'beam2', None, self.deck_width/8, None, 0, 0, None)
           
        ut.Duplicate(member.beam, 'beam3', None, -self.deck_width/8, None, 0,0, None)

        ut.Duplicate(member.beam, 'beam4', None, -3*self.deck_width/8, None, 0, 0, None)
         

        member.beam.select_set(True)
        bpy.ops.object.delete()


    def floorbeam(self):
        
        member = cp.Floorbeam(self.deck_thickness, self.deck_height, 'floorbeam', self.truss_width, self.truss_thickness,self.truss_height) 

        for i in range(self.truss_number):
            # dis = self.truss_thickness/2 + 0.5
            
            dis = self.deck_width/4

            theta = float(self.fyd(int(i*self.truss_distance),self.a,self.b))

            ut.Duplicate(member.floorbeam, 'floorbeam1'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.truss_distance),self.a,self.b) + dis*np.cos(theta), None, 0, 0, np.arctan(theta))

            ut.Duplicate(member.floorbeam, 'floorbeam2'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b),\
                self.fy(int(i*self.truss_distance),self.a,self.b), None, 0, 0, np.arctan(theta))

            ut.Duplicate(member.floorbeam, 'floorbeam3'+str(i+1), self.fx(int(i*self.truss_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.truss_distance),self.a,self.b) - dis*np.cos(theta), None, 0, 0, np.arctan(theta))

        member.floorbeam.select_set(True)
        bpy.ops.object.delete()

    def column(self):

        #self.column_height -= self.bearing_thickness
        member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length,  self.deck_height, self.deck_thickness, 'pi column')
        member.piColumn()        
        for i in range(self.column_number): 

            theta = float(self.fyd(int(i*self.column_distance),self.a,self.b))
            print('theta = ' + str(theta))
            
            ut.Duplicate(member.column, 'Pi_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(theta))            

        member.column.select_set(True)
        bpy.ops.object.delete()
        #self.column_height += self.bearing_thickness


    def bearing(self):
        a = self.beam_width
        T = self.bearing_thickness
        t = T/3
        d = a

        member = cp.ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
            dis = a/2 + 0.5

            theta = float(self.fyd(i*self.column_distance,self.a,self.b))      
        
            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta)*2,\
                 self.fy(int(i*self.column_distance),self.a,self.b) + 3*self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))

            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) + self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))
        
            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) -self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))

            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta)*2,\
                self.fy(int(i*self.column_distance),self.a,self.b) -3*self.deck_width/8, self.deck_height - self.deck_thickness - self.beam_height - self.bearing_thickness/2,\
                     0, math.pi/2, np.arctan(theta))
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()

class SlabBridge:
    def __init__(self, l_deck = 500, h_deck = 11, t_deck = 1, w_deck = 10, d_column = 10, l_column = 2, w_column = 6, h_column = 10):
        
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_height = h_column
        self.column_length = l_column
        self.column_width = w_column
        self.column_distance = d_column

        self.bearing_thickness = 0.3

        self.column_number = int(self.deck_length/self.column_distance)-1 # 49
        self.column_thickness = 1

        index_deck_list = [1,2,3,4,5,6]
        self.deck_index = random.choice(index_deck_list)

        self.a = (np.random.rand())*20
        self.b = (np.random.rand())*1000

        self.theta = np.array([np.arctan(float(self.fyd(i,self.a,self.b))) for i in range(1,self.deck_length+1)])


        self.column()
        self.deck()
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

    def column(self):
        Span = self.column_distance
        Length = self.deck_length
        name = 'column'

        self.column_height -= self.bearing_thickness
        member = cp.Column(self.column_width, self.column_height, self.column_thickness, self.column_length,  self.deck_height, self.deck_thickness, 'pi column')
        member.piColumn()        
        for i in range(self.column_number): 

            ut.Duplicate(member.column, 'Pi_column'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b),\
                        self.fy(int(i*self.column_distance),self.a,self.b), None, 0, 0, np.arctan(float(self.fyd(int(i*self.column_distance),self.a,self.b))))            

        member.column.select_set(True)
        bpy.ops.object.delete()
        self.column_height += self.bearing_thickness


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
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete solid deck', t, quat, self.deck_width, None)
            member.concrete_solid()
        
        elif self.deck_index == 2:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete PK deck', t, quat, self.deck_width, None)
            member.concrete_PK()
        
        elif self.deck_index == 3:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete box deck', t, quat, self.deck_width, None)
            member.concrete_box()
        
        elif self.deck_index == 4:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'comcrete constalia deck', t, quat, self.deck_width, None)
            member.concrete_costalia()

        elif self.deck_index == 5:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel box deck', t, quat, self.deck_width, None)
            member.steel_box() 

        elif self.deck_index == 6:
            member = cp.Slab(self.deck_thickness, self.deck_height, self.deck_length, self.column_width, self.column_thickness, 'steel sidebox deck', t, quat, self.deck_width, None)
            member.steel_sidebox()       
    
    def bearing(self):
        a = self.column_length
        T = self.bearing_thickness
        t = T/3
        d = a

        member = cp.ColumnBearing(a, d, T, t, "bearing")
        
        for i in range(self.column_number):
            dis = a/2 + 0.5

            theta = float(self.fyd(i*self.column_distance,self.a,self.b))
            # member1 = ColumnBearing(a, d, T, t, "bearing" + str(i*2+1))
            # member1.column_bearing.rotation_euler[2] = np.arctan(theta)
            # member1.column_bearing.location.x = self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta)
            # member1.column_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            # member1.column_bearing.location.y = self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta)

            # member2 = ColumnBearing(a, d, T, t, "bearing" + str(i*2+2))
            # member2.column_bearing.rotation_euler[2] = np.arctan(theta) 
            # member2.column_bearing.location.x = self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta)
            # member2.column_bearing.location.z = self.deck_height - self.bearing_thickness/2 - self.deck_thickness
            # member2.column_bearing.location.y = self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta)  

        
            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) - dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) + dis*np.cos(theta), self.deck_height - self.bearing_thickness/2 - self.deck_thickness,\
                     0, math.pi/2, np.arctan(theta))

            ut.Duplicate(member.column_bearing, 'bearing'+str(i+1), self.fx(int(i*self.column_distance),self.a,self.b) + dis*np.sin(theta),\
                self.fy(int(i*self.column_distance),self.a,self.b) - dis*np.cos(theta), self.deck_height - self.bearing_thickness/2 - self.deck_thickness,\
                     0, math.pi/2, np.arctan(theta))
        
        member.column_bearing.select_set(True)
        bpy.ops.object.delete()
        