import bpy
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from sympy import *
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

Bg = bpy.data.texts["bridge.py"].as_module() 
###########
'''
l_deck = 100
h_deck = 10 
t_deck = 1
w_deck = 10
index_deck = 1


d_column = 0
l_column = 1.5 
w_column = 10
w_column2 = 1.5
span_column = 10
distant_column = 5 # distance between two columns in one layer (y-axis)
h_column = 36
index_column = random.choice([0,1])



l_bearing = 1
t_bearing = random.choice([0,0.15,0.2])
w_bearing = 0.6

t_piercap = 1
index_piercap = random.choice([0,1])
w_girder = random.choice([0.6,0])
t_girder = w_girder

w_parapet = 0.2
t_parapet = 1

mirror = int(np.random.rand()*2)+1


# good_luck = Bg.BeamBridge()
'''

###########


l_deck = 20.
h_deck = 8. 
t_deck = 1. 
w_deck = 10.
d_column = 0.
l_column = 2. 
w_column = 10. 
h_column = 36.

good_luck = Bg.CableStayedBridge(l_deck, h_deck, t_deck, w_deck, d_column, l_column, w_column, h_column)

'''
###########
h_column = np.random.rand()*10+30

t_column = 1

l_column = 2

w_column = np.random.rand()*5+10

h_deck = h_column/4

t_deck = 1

index_column_list = [2,3]
index_column = random.choice(index_column_list)

if index_column == 1 or index_column ==2 or index_column == 3:
    index_deck_list = [1,2,4,5,6]
    face_cable = 2
elif index_column == 4:
    index_deck_list = [3]
    face_cable = 1
index_deck = random.choice(index_deck_list)


bridge = Bg.SuspensionBridgeGenerator(0, h_deck, t_deck, 10, 0, l_column, w_column, h_column)

###########
good_luck = Bg.DeckArchBridge()

###########
Bg.SteelGirderBridge()   

###########
Bg.SlabBridge()          
'''