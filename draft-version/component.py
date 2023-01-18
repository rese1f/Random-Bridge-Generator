import bpy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *
import math


cf = bpy.data.texts["cfg.py"].as_module()
mb = bpy.data.texts["member.py"].as_module()
ut = bpy.data.texts["utils.py"].as_module()

class SuperStructure:
    def __init__(self):
        self.cable = None
        self.parapet = None
        self.rail_slab = None
        self.rail_sleeper = None
        self.rail_track = None

class Cable(SuperStructure):
    def __init__(self, cable_start, cable_end, name, cable_radius = 0.05, num = 8):
        super().__init__()

        self.cable_start = cable_start
        self.cable_end = cable_end
        self.cable_radius = cable_radius
        self.cable_fit_num = num
        self.name = name

        cfg_start = cf.setCircle(self.cable_start, self.cable_radius, self.cable_fit_num)
        cfg_end = cf.setCircle(self.cable_end, self.cable_radius, self.cable_fit_num)

        member = mb.Circle(cfg_start, cfg_end, 1, True, None, None)
        member.createObj(name)
        self.cable = member.obj

class Cable2d(SuperStructure):
    def __init__(self, name, length, trans = None, cable_radius = 0.1):
        super().__init__()

        self.cable_radius = cable_radius
        self.length = length
        self.trans = trans #transform

        cfg = cf.setCircle(None, self.cable_radius, None)
        
        member = mb.Circle2d(cfg, self.length, self.trans, None,name)

        member.createObj(name)
        self.cable_main = member.obj
        
class Cable2d_s(SuperStructure):
    def __init__(self, name, length, cable_radius = 0.05):
        super().__init__()  
        self.length = length
        self.cable_radius = cable_radius
        cfg = cf.setCircle(None, self.cable_radius, None)
        
        t = np.zeros((self.length, 3))
        t[:,0] = np.arange(self.length)
        member = mb.Circle2d(cfg, self.length, t, None,name)
        member.createObj(name)

        self.cable = member.obj

class Parapet(SuperStructure):
    def __init__(self, w_deck, l_deck, h_deck, t_deck, w_parapet, t_parapet,name, tran, quat):
        super().__init__()
        self.deck_width = w_deck
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck

        self.parapet_width = w_parapet
        self.parapet_thickness = t_parapet

        self.tran = tran
        self.quat = quat

        self.name = name

    def rectangleParapet(self):
        cfg_start = cf.setRectangle(self.parapet_width,self.parapet_thickness+self.deck_thickness/2,\
            self.parapet_thickness+self.deck_height)
        cfg_end = cfg_start

        member = mb.Rectangle(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)

        self.parapet = member.obj

class Railway(SuperStructure):
    def __init__(self, w_deck, l_deck, h_deck, t_deck, w_railslab, t_railslab,\
        w_rail, t_rail, h_sleeper, l_sleeper, span_sleeper, name, tran, quat):
        self.deck_width = w_deck
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        
        self.railslab_width = w_railslab
        self.railslab_thickness = t_railslab

        self.rail_width = w_rail
        self.rail_thickness = t_rail

        self.sleeper_height = h_sleeper
        self.sleeper_length = l_sleeper
        self.Span_sleeper = span_sleeper

        self.tran = tran
        self.quat = quat

        self.name = name

    def railSlabBuild(self):
        cfg_start = cf.setRectangle(self.railslab_width,self.railslab_thickness,\
            self.railslab_thickness+self.deck_height)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        
        self.rail_slab = member.obj

    def railSleeperBuild(self):
        cfg_start = cf.setRectangle(self.railslab_width*0.8,self.railslab_thickness,\
            self.railslab_thickness+self.deck_height+self.sleeper_height)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start, cfg_end, self.sleeper_length)
        
        member.createObj(self.name)
        # print(self.name)
        
        self.rail_sleeper = member.obj
    
    def railBuild(self):
        cfg_start = cf.setRectangle(self.rail_width,self.railslab_thickness,\
            self.railslab_thickness*2+self.deck_height+self.sleeper_height)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)

        self.rail_track = member.obj

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

        cfg_start = cf.setArchringBasic(self.archring_width, self.archring_height, self.archring_thickness)
        cfg_end = cfg_start


        member = mb.ArchRing(cfg_start, cfg_end, self.archring_length, False, self.t, self.quat)
        member.createObj(self.name)
        self.archring = member.obj

class Beam(SuperStructure):
    def __init__(self, w_beam, h_beam, l_beam, t_webplate, t_flange, h_deck, t_deck, name, tran = None, quat = None):
        super().__init__()
        
        self.beam_width = w_beam
        self.beam_height = h_beam
        self.beam_length = l_beam
        self.webplate_thickness = t_webplate
        self.flange_thickness = t_flange
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.tran = tran
        self.quat = quat

        self.name = name
    

        cfg_start = cf.setBeamBasic(self.beam_width, self.beam_height, self.webplate_thickness, self.flange_thickness, self.deck_height, self.deck_thickness)
        cfg_end = cfg_start

        member = mb.IBeam(cfg_start, cfg_end, self.beam_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.beam = member.obj

class Floorbeam(SuperStructure):
    def __init__(self, t_deck, h_deck, name, w_truss, t_truss, h_truss):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        
        self.name = name
        self.truss_width = w_truss
        self.truss_thickness = t_truss 
        self.truss_height = h_truss
   

        t_h = 0.1
        t_v = 0.1
        
        
        b11 = -self.truss_width/2 + t_v
        c11 = self.deck_height - self.deck_thickness - self.truss_height/2 - 2*self.truss_height/5
        b12 = -self.truss_width/2 + t_v + 2*self.truss_width/5
        c12 = self.deck_height - self.deck_thickness - self.truss_height/2
        b13 = -self.truss_width/2 + t_v
        c13 = self.deck_height - self.deck_thickness - self.truss_height/2 + 2*self.truss_height/5
        
        b21 = self.truss_width/2 - t_v
        c21 = self.deck_height - self.deck_thickness - self.truss_height/2 - 2*self.truss_height/5
        b22 = self.truss_width/2 - t_v
        c22 = self.deck_height - self.deck_thickness - self.truss_height/2 + 2*self.truss_height/5
        b23 = self.truss_width/2 - t_v - 2*self.truss_width/5
        c23 = self.deck_height - self.deck_thickness - self.truss_height/2
        
        b31 = 2*self.truss_width/5
        c31 = self.deck_height - self.deck_thickness - self.truss_height + t_h
        b32 = 0
        c32 = self.deck_height - self.deck_thickness - self.truss_height + t_h + 7*self.truss_height/20
        b33 = -2*self.truss_width/5
        c33 = self.deck_height - self.deck_thickness - self.truss_height + t_h
        
        b41 = 0
        c41 = self.deck_height - self.deck_thickness - t_h - 7*self.truss_height/20
        b42 = 2*self.truss_width/5
        c42 = self.deck_height - self.deck_thickness - t_h
        b43 = -2*self.truss_width/5
        c43 = self.deck_height - self.deck_thickness - t_h
        
        
        cfg_start = cf.setRectangle(self.truss_width, self.truss_height, self.deck_height - self.deck_thickness)  
        cfg_end = cfg_start

        orig = mb.Rectangle(cfg_start, cfg_end, self.truss_thickness)
        orig.createObj(self.name)
        
        name = 'left_hollow'
        cfg1_start = cf.setTriangle(1, b11, c11, b12, c12, b13, c13) 
        cfg1_end = cf.setTriangle(-1, b11, c11, b12, c12, b13, c13)
        tria1 = mb.Triangle(cfg1_start, cfg1_end, 1)
        tria1.createObj(name)
        ut.Hollow(orig.obj, tria1.obj)        

        name = 'right_hollow' 
        cfg2_start = cf.setTriangle(1, b21, c21, b22, c22, b23, c23) 
        cfg2_end = cf.setTriangle(-1, b21, c21, b22, c22, b23, c23)
        tria2 = mb.Triangle(cfg2_start, cfg2_end, 1)
        tria2.createObj(name)
        ut.Hollow(orig.obj, tria2.obj)       
       
        name = 'down_hollow' 
        cfg3_start = cf.setTriangle(2, b31, c31, b32, c32, b33, c33) 
        cfg3_end = cf.setTriangle(-2, b31, c31, b32, c32, b33, c33)
        tria3 = mb.Triangle(cfg3_start, cfg3_end, 1)
        tria3.createObj(name)
        ut.Hollow(orig.obj, tria3.obj)     
        
        name = 'up_hollow'
        cfg4_start = cf.setTriangle(1, b41, c41, b42, c42, b43, c43) 
        cfg4_end = cf.setTriangle(-2, b41, c41, b42, c42, b43, c43)
        tria4 = mb.Triangle(cfg4_start, cfg4_end, 1)
        tria4.createObj(name)
        ut.Hollow(orig.obj, tria4.obj) 


        self.floorbeam = orig.obj




class SubStructure:
    def __init__(self):
        self.column = None
        self.girder_lateral = None
        self.girder_logitude = None

class Column(SubStructure):
    def __init__(self, w_column, h_column, t_column, l_column, h_deck_t, t_deck, name):
        super().__init__()
        self.column_width = w_column
        self.column_height = h_column
        self.column_thickness = t_column # latitude
        self.column_length = l_column # longitude
        self.deck_height = h_deck_t
        self.deck_thickness = t_deck
        self.name = name
        self.cable_function = [0, 0, 0, 0, 0, 0]

    def A1(self):
        cfg_start = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start

        h_deck = self.deck_height - self.deck_thickness

        k = self.column_height / (self.column_width/2)
        b_out = self.column_height
        b_in = self.column_height - k * self.column_thickness
        b_cable = (b_out + b_in) / 2

        w_tria = -2 * (h_deck - b_in) / k
        h_tria = self.column_height - self.column_thickness * k - h_deck
        H_tria = h_deck + h_tria  

        cfg_hollow_start = cf.setTriangle2(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start

        orig = mb.A1(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = mb.Triangle2(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)
        hollow.createObj('hollow')

        ut.Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = self.column_height - self.column_thickness * k / 2 ## [z_start, z_end, k, b, b_in]         

    def double(self):
        cfg_start = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start

        h_deck = self.deck_height - self.deck_thickness
        
        member = mb.Double(cfg_start, cfg_end, self.column_length)
        member.createObj(self.name)

        self.column = member.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.column_width/2 - self.column_thickness/2
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = h_deck + (self.column_height - h_deck) * 9/10

    def door(self):
        cfg_start = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness        

        w_rec = self.column_width - 2 * self.column_thickness
        h_rec = self.column_height - 3 * self.column_thickness - h_deck
        H_rec = h_deck + h_rec

        cfg_hollow_start = cf.setRectangle(w_rec, h_rec, H_rec)
        cfg_hollow_end = cfg_hollow_start

        h_deck = self.deck_height - self.deck_thickness

        orig = mb.Door(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = mb.Rectangle(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)        
        hollow.createObj('hollow')

        ut.Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = 0
        self.cable_function[3] = 0
        self.cable_function[4] = 0
        self.cable_function[5] = self.column_width/2 - self.column_thickness/2
        self.cable_function[0] = h_deck + (self.column_height - h_deck) / 3
        self.cable_function[1] = h_deck + (self.column_height - h_deck) * 9/10 
         
    def tower(self):
        cfg_start = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness

        a = random.uniform(1/5, 1/3)
        h_cable = a * self.column_height
        b = random.uniform(0.5, 0.8)
        w_bottom = b * self.column_width
        w_top = 2 * self.column_thickness
        
        k = (self.column_height - h_cable - h_deck) / (self.column_width/2 - self.column_thickness) 
        b_cable = 0
        b_in = self.column_height - h_cable

        w_tria = self.column_width - 2 * self.column_thickness
        h_tria = self.column_height - h_cable - h_deck
        H_tria = h_deck + h_tria
        cfg_hollow_start = cf.setTriangle2(w_tria, h_tria, H_tria)
        cfg_hollow_end = cfg_hollow_start
         
        orig = mb.Tower(cfg_start, cfg_end, self.column_length)
        orig.createObj(self.name)
        hollow = mb.Triangle2(cfg_hollow_start, cfg_hollow_end, self.column_length + 5)
        hollow.createObj('hollow')

        ut.Hollow(orig.obj, hollow.obj)

        self.column = orig.obj

        self.cable_function[2] = k
        self.cable_function[3] = b_cable
        self.cable_function[4] = b_in
        self.cable_function[5] = 0.2
        self.cable_function[0] = self.column_height - h_cable * 9/10
        self.cable_function[1] = self.column_height - h_cable * 1/10 

    def rectangle_column(self):
        cfg_start = cf.setRectangle(self.column_width, self.column_height, self.column_height)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start,cfg_end,self.column_length)
        member.createObj(self.name)
        self.column = member.obj
    
    def cylinder_column(self):
        # cfg = setCircle(None,self.column_width/2,None)
        # member = Circle2d(cfg,self.column_height, None, None)
        # member.createObj(self.name)
        # member.obj.rotation_euler[1] = -math.pi/2
        # self.column = member.obj
        cfg_start = cf.setCircle([0,0,0],self.column_width/2,50)
        cfg_end = cf.setCircle([self.column_height,0,0],self.column_width/2,50)
        member = mb.Circle(cfg_start, cfg_end, 1, True)
        member.createObj(self.name)
        member.obj.rotation_euler[1] = -math.pi/2
        self.column = member.obj

    def piColumn(self):
        cfg_start = cf.setColumnBasic(self.column_width, self.column_height, self.column_thickness, self.deck_height, self.deck_thickness) 
        cfg_end = cfg_start
        h_deck = self.deck_height - self.deck_thickness
        column = mb.PiColumn(cfg_start, cfg_end, self.column_length)
        column.createObj(self.name)
        self.column = column.obj

# a = Column(10, 36, 1, 3, 8, 1, 'column')
# a.double()
# a.column.location.x = 5
# a.column.location.y = 15 

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

  
        cfg_start = cf.setRectangle(self.column_width, self.column_height, self.column_height)
        cfg_end = cfg_start

   
        member = mb.Rectangle(cfg_start, cfg_end, self.column_length)
        member.createObj(self.name)
        self.uprightcolumn = member.obj              
                
                
    def duplicateDistance(self, w, l):
        n = int(l / w)
        D = np.zeros(n)
        for i in range(n):
            D[i] = w / 2 + w * i
        return D

class Pier(SubStructure):
    def __init__(self, w_pier, h_pier, t_pier, name):
        super().__init__()
        self.pier_width = w_pier
        self.pier_height = h_pier
        self.pier_thickness = t_pier 
        self.name = name

  
        cfg_start = cf.setRectangle(self.pier_width, self.pier_height, self.pier_height)
        cfg_end = cfg_start

   
        member = mb.Rectangle(cfg_start, cfg_end, self.pier_thickness)
        member.createObj(self.name)
        self.pier = member.obj              
        


class Girder(SubStructure):
    def __init__(self, w_girder=0.6, t_girder=0.6, l_deck=10, h_deck=10, t_deck=1, w_deck=10, l_column=1.5, t_bearing=0.2, w_bearing=0.6, name='girder', tran = None, quat = None):
        super().__init__()
        self.girder_width = w_girder
        self.girder_thickness = t_girder
        self.deck_length = l_deck
        self.deck_height = h_deck
        self.deck_thickness = t_deck
        self.deck_width = w_deck
        self.column_length = l_column
        self.bearing_thickness = t_bearing
        self.bearing_width = w_bearing
        self.name = name
        
        self.tran = tran
        self.quat = quat

        self.piercap_thickness = 1
        self.bearing_length = 1
        self.column2_distance = 5
        self.W_column2 = 1.5




    def I_girder(self):
        cfg_start = cf.setI_Beam(self.girder_width,self.girder_thickness,\
        self.deck_height-self.deck_thickness,self.girder_thickness/10,self.girder_width/3)
        cfg_end = cfg_start
        member = mb.I_Beam(cfg_start,cfg_end,self.deck_length,False,self.tran,self.quat)

        member.createObj(self.name)
        self.girder_logitude = member.obj

    def rectangle_girder(self):
        cfg_start = cf.setRectangle(self.girder_width/2, self.girder_thickness,\
        self.deck_height-self.deck_thickness)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start,cfg_end,self.deck_length,False,self.tran,self.quat)

        member.createObj(self.name)
        self.girder_logitude = member.obj


    def rectangle_pier_cap(self):
        cfg_start = cf.setPierCap(self.deck_width,\
            self.deck_height-self.deck_thickness-self.girder_thickness-self.bearing_thickness,1,1,1)
        cfg_end = cfg_start
        member = mb.PierCap(cfg_start,cfg_end,self.column_length)

        member.createObj(self.name)
        self.girder_lateral = member.obj

    def box_pier_cap(self):
        cfg_start = cf.setBoxPierCap(self.deck_width,(self.deck_width+self.column2_distance*2)/3,\
            self.column2_distance,\
            self.deck_height-self.deck_thickness-self.girder_thickness-self.bearing_thickness,0.2,\
                self.piercap_thickness-0.2)
        cfg_end = cfg_start
        member = mb.BoxPierCap(cfg_start,cfg_end,self.column_length)

        member.createObj(self.name)
        self.girder_lateral = member.obj

# a = Girder()
# a.box_pier_cap()

class Deck:
    def __init__(self):
        self.slab = None
    
class Slab(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_column, t_column, name,\
         tran = None, quat = None, w_deck = None, cable_function = None):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        self.deck_length = l_deck
        self.column_width = w_column
        self.column_thickness = t_column
        self.truss_thickness = None

        self.tran = tran
        self.quat = quat

        self.name = name
        self.cable_function = cable_function

        if cable_function == None:
            self.deck_width = w_deck
        else:
            k = self.cable_function[2]
            b_in = self.cable_function[4]
            h = self.deck_height + self.deck_thickness
            if b_in == 0:
                self.deck_width = self.column_width - self.column_thickness * 2
            else:
                self.deck_width = - (h - b_in)/k * 2
    
    def concrete_solid(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteSolid(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
 

    def concrete_PK(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcretePK(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj


    def concrete_box(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
   

    def concrete_costalia(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteCostalia(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
    

    def steel_box(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.SteelBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
     

    def steel_sidebox(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.SteelSidebox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj

    def truss(self):
        T_truss = 5
        self.truss_thickness = T_truss
        h = self.deck_height - self.deck_thickness + T_truss
        if self.cable_function != None:
            k = self.cable_function[2]
            b_in = self.cable_function[4]
            if b_in == 0:
                self.deck_width = self.column_width - self.column_thickness * 2
            else:
                self.deck_width = - (h - b_in)/k * 2
        
        v_width = 0.25
        h_width = 0.25

        cfg_start = cf.setRectangle(self.deck_width, self.truss_thickness, h)  
        cfg_end = cfg_start

        cfg_hollow_start = cf.setRectangle(self.deck_width - 2*h_width, self.truss_thickness - 2*v_width, h - v_width) 
        cfg_hollow_end = cfg_hollow_start

        orig = mb.Rectangle(cfg_start, cfg_end, self.deck_length)
        orig.createObj(self.name)
        hollow = mb.Rectangle(cfg_hollow_start, cfg_hollow_end, self.deck_length + 5)
        hollow.createObj('hollow') 

        ut.Hollow(orig.obj, hollow.obj)

        thick_bar = v_width * 1.5
        l = self.deck_length - thick_bar
        width_bar = 4
        height_bar = T_truss

        a11 = -l/2 + thick_bar/2
        b11 = thick_bar + self.deck_height - self.deck_thickness
        a12 = -l/2 + width_bar - thick_bar/2
        b12 = thick_bar+ self.deck_height - self.deck_thickness
        a13 = -l/2 + thick_bar/2
        b13 = height_bar - thick_bar -  height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness

        a23 = -l/2 + thick_bar/2
        b23 = height_bar - thick_bar + self.deck_height - self.deck_thickness
        a22 = -l/2 + width_bar - thick_bar/2
        b22 = height_bar - thick_bar + self.deck_height - self.deck_thickness
        a21 = -l/2 + width_bar - thick_bar/2 
        b21 = thick_bar + height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness    

        for i in range(int(l/width_bar)):
        # for i in range(1):    
            name = 'down_hollow' + str(i)
            cfg1_start = cf.setTriangle(a11 + width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, 10) 
            cfg1_end = cf.setTriangle(a11+ width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, -10)
            tria1 = mb.Triangle2(cfg1_start, cfg1_end, 1)
            tria1.createObj(name)
            ut.Hollow(orig.obj, tria1.obj)

        for i in range(int(l/width_bar)):
        # for i in range(1):
            name = 'up_hollow' + str(i)
            cfg2_start = cf.setTriangle(a21 + width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, 10) 
            cfg2_end = cf.setTriangle(a21+ width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, -10)
            tria2 = mb.Triangle2(cfg2_start, cfg2_end, 1)
            tria2.createObj(name)
            ut.Hollow(orig.obj, tria2.obj)             

        self.slab = orig.obj

# sample  = Slab(2,1,1,1,1,'sample',None,None,8)
# sample.concrete_solid()

class Slab_smy(Deck):
    def __init__(self, t_deck, h_deck, l_deck, w_deck, name, \
        tran = None, quat = None):
        super().__init__()

        self.deck_thickness = t_deck
        self.deck_height = h_deck
        self.deck_length = l_deck
        self.deck_width = w_deck
     
        self.truss_thickness = None

        self.tran = tran
        self.quat = quat

        self.name = name

    def concrete_solid(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteSolid(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
  

    def concrete_PK(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcretePK(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj


    def concrete_box(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
    

    def concrete_costalia(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.ConcreteCostalia(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
     

    def steel_box(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.SteelBox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
      

    def steel_sidebox(self):
        cfg_start = cf.setDeckBasic(self.deck_width, self.deck_thickness, self.deck_height)
        cfg_end = cfg_start

        member = mb.SteelSidebox(cfg_start, cfg_end, self.deck_length, False, self.tran, self.quat)
        member.createObj(self.name)
        self.slab = member.obj
         
        
'''
    # def truss(self):
    #     T_truss = 5
    #     self.truss_thickness = T_truss
    #     h = self.deck_height - self.deck_thickness + T_truss
    #     if self.cable_function != None:
    #         k = self.cable_function[2]
    #         b_in = self.cable_function[4]
    #         if b_in == 0:
    #             self.deck_width = self.column_width - self.column_thickness * 2
    #         else:
    #             self.deck_width = - (h - b_in)/k * 2
        
    #     v_width = 0.25
    #     h_width = 0.25

    #     cfg_start = setRectangle(self.deck_width, self.truss_thickness, h)  
    #     cfg_end = cfg_start

    #     cfg_hollow_start = setRectangle(self.deck_width - 2*h_width, self.truss_thickness - 2*v_width, h - v_width) 
    #     cfg_hollow_end = cfg_hollow_start

    #     orig = Rectangle(cfg_start, cfg_end, self.deck_length)
    #     orig.createObj(self.name)
    #     hollow = Rectangle(cfg_hollow_start, cfg_hollow_end, self.deck_length + 5)
    #     hollow.createObj('hollow') 

    #     Hollow(orig.obj, hollow.obj)

    #     thick_bar = v_width * 1.5
    #     l = self.deck_length - thick_bar
    #     width_bar = 4
    #     height_bar = T_truss

    #     a11 = -l/2 + thick_bar/2
    #     b11 = thick_bar + self.deck_height - self.deck_thickness
    #     a12 = -l/2 + width_bar - thick_bar/2
    #     b12 = thick_bar+ self.deck_height - self.deck_thickness
    #     a13 = -l/2 + thick_bar/2
    #     b13 = height_bar - thick_bar -  height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness

    #     a23 = -l/2 + thick_bar/2
    #     b23 = height_bar - thick_bar + self.deck_height - self.deck_thickness
    #     a22 = -l/2 + width_bar - thick_bar/2
    #     b22 = height_bar - thick_bar + self.deck_height - self.deck_thickness
    #     a21 = -l/2 + width_bar - thick_bar/2 
    #     b21 = thick_bar + height_bar/width_bar * thick_bar + self.deck_height - self.deck_thickness    

    #     for i in range(int(l/width_bar)):
    #     # for i in range(1):    
    #         name = 'down_hollow' + str(i)
    #         cfg1_start = setTriangle2(a11 + width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, 10) 
    #         cfg1_end = setTriangle2(a11+ width_bar*i, b11, a12+ width_bar*i, b12, a13+ width_bar*i, b13, -10)
    #         tria1 = Triangle2(cfg1_start, cfg1_end, 1)
    #         tria1.createObj(name)
    #         Hollow(orig.obj, tria1.obj)

    #     for i in range(int(l/width_bar)):
    #     # for i in range(1):
    #         name = 'up_hollow' + str(i)
    #         cfg2_start = setTriangle2(a21 + width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, 10) 
    #         cfg2_end = setTriangle2(a21+ width_bar*i, b21, a22+ width_bar*i, b22, a23+ width_bar*i, b23, -10)
    #         tria2 = Triangle2(cfg2_start, cfg2_end, 1)
    #         tria2.createObj(name)
    #         Hollow(orig.obj, tria2.obj)             

    #     self.slab = orig.obj
'''


# b = Slab(1, 8, 50, 10, 1, 'slab', 5)
# b.concrete_costalia()
# b.slab.location.x = 0

class Bearing:
    def __init__(self):
        self.cable_base = None
        self.cable_top = None
        self.column_bearing = None
        self.girder_bearing = None

class CableBase(Bearing):
    def __init__(self, t1, t2, t3, r2, name, turn = 1):                         
        cfg_start = cf.setCircle([(0 - t3*2/3) * turn, 0, 0], r2, 50)
        cfg_end = cf.setCircle([(t3*1/3 + t2) * turn, 0, 0], r2, 50)
        member = mb.Circle(cfg_start, cfg_end, 1)
        member.createObj(name)
        self.cable_base = member.obj

class CableTop(Bearing):
    def __init__(self, t1, t2, t3, r1, name, turn = 1):                         
        cfg_start = cf.setCircle([-(t3*3/4 + t2) * turn, 0, 0], r1, 50)
        cfg_end = cf.setCircle([-(0 - t3/4) * turn, 0, 0], r1, 50)
        member = mb.Circle(cfg_start, cfg_end, 1)
        member.createObj(name)
        self.cable_base = member.obj   

class DeckBearing(Bearing):
    def __init__(self, a, d, T, t, name):
        cfg_start = cf.setRectangle(a, (T-t)/2, (T-t)/2+t/2)
        cfg_end = cfg_start
        rec1 = mb.Rectangle(cfg_start, cfg_end, a) 
        rec1.createObj('rec1')
        
        cfg_start = cf.setRectangle(a, (T-t)/2, -t/2)
        cfg_end = cfg_start
        rec2 = mb.Rectangle(cfg_start, cfg_end, a) 
        rec2.createObj('rec2')
        
        cfg_start = cf.setCircle([-t/2, 0, 0], d/2, 50)
        cfg_end = cf.setCircle([t/2, 0, 0], d/2, 50)
        deck_bearing = mb.Circle(cfg_start, cfg_end, 1)
        deck_bearing.createObj(name)
        deck_bearing.obj.rotation_euler[1] = math.pi/2
        
        ut.Merge([deck_bearing.obj, rec1.obj, rec2.obj])
        self.deck_bearing = deck_bearing.obj

class ColumnBearing(Bearing):
    def __init__(self, a, d, T, t, name):
        cfg_start = cf.setRectangle(a, (T-t)/2, (T-t)/2+t/2)
        cfg_end = cfg_start
        rec1 = mb.Rectangle(cfg_start, cfg_end, a) 
        rec1.createObj('rec1')
        
        cfg_start = cf.setRectangle(a, (T-t)/2, -t/2)
        cfg_end = cfg_start
        rec2 = mb.Rectangle(cfg_start, cfg_end, a) 
        rec2.createObj('rec2')
        
        cfg_start = cf.setCircle([-t/2, 0, 0], d/2, 50)
        cfg_end = cf.setCircle([t/2, 0, 0], d/2, 50)
        column_bearing = mb.Circle(cfg_start, cfg_end, 1)
        column_bearing.createObj(name)
        column_bearing.obj.rotation_euler[1] = math.pi/2
        
        ut.Merge([column_bearing.obj, rec1.obj, rec2.obj])
        self.column_bearing = column_bearing.obj

class GirderBearing(Bearing):
    def __init__(self, w_deck=10, h_deck=8, t_deck=1, t_bearing=0.5, l_bearing=1, t_girder=1, name='gird_bearing'):    
        cfg_start = cf.setRectangle(w_deck*0.9, t_bearing, h_deck-t_deck-t_girder)
        cfg_end = cfg_start
        member = mb.Rectangle(cfg_start,cfg_end,l_bearing)
        member.createObj(name)
        self.girder_bearing = member.obj    

