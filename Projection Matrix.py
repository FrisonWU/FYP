import cv2
import numpy as np
import math
import sympy
from sympy import *
#a = symbols('a',communtative=True)
# This projection matrix is based on perspective projection matrix
pixel_x = 480
pixel_y = 640
span_x = 199# 280mm
span_y = 375# 177mm
D = 265 # Camera distance z = -475mm
roll = 30
roll = roll*math.pi/180
pitch = 80
pitchrad = pitch*math.pi/180

roll_rotate = np.mat([[1,0,0,0],[0,math.cos(roll),-math.sin(roll),0],[0,math.sin(roll),math.cos(roll),0],[0,0,0,1]])
ptich_rotate = np.mat([[math.cos(pitchrad),0,math.sin(pitchrad),0],[0,1,0,0],[-math.sin(pitchrad),0,math.cos(pitchrad),0],[0,0,0,1]])
Wide_Input = np.mat([[0,0,D/2,1],[-(span_x/2),0,D/2,1],[span_x/2,0,D/2,1]])
Projection_Matrix = np.mat([[1,0,0,0],[0,1,0,0],[0,0,1,1/D],[0,0,-D,1]])
Pixel_Transform_Matrix = np.mat([[pixel_x/span_x,0,0,0],[0,pixel_y/span_y,0,0],[0,0,1,0],[pixel_x/2,pixel_y/2,0,1]])
#Pf= np.array([[(1/r)*sympy.cot(a/2),0,0,0],[0,sympy.cot(a/2),0,0],[0,0,f/(f-n),(-f*n)/(f-n)],[0,0,1,0]])    #定义一个数组
print(Wide_Input*Projection_Matrix)

Whole_Matrix = Projection_Matrix*Pixel_Transform_Matrix
print (Whole_Matrix)
print(np.dot(Projection_Matrix,Pixel_Transform_Matrix))
#print (Pixel_Transform_Matrix)

# print (Whole_Matrix)
Input = np.array([[10,10,0,1]]).T
Output = np.dot(Whole_Matrix.T,Input)
# Output_homo = Output/Output[0,3]
# Yaw_homo = Output_homo*roll_rotate
# Pitch_homo = Output_homo*ptich_rotate
print(Output)
# print (Output_homo)
# print(Yaw_homo)
# print(Pitch_homo)