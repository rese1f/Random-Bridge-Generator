#import bpy
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
import torch
import math
from sympy import *

class total_func:
    def __init__(self,fy):
        self.fy = fy
        #self.fz = fz

    def fyd(self):
        """
        goal: first order derivative of fy

        """
        k=Symbol("k")
        fyd=diff(fy(k),k,1)
        fyd=fyd.subs(k,x)
        return fyd

def fy(x):                                
    return x**2/100
a= total_func(fy) 
