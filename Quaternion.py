from sympy import *
from sympy.abc import x,y
import numpy as np
import math
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
q1 = Symbol('q1')
q2 = Symbol('q2')
q3 = Symbol('q3')
q4 = Symbol('q4')
a,b,c,d = symbols('a,b,c,d')

Input_Matix = Matrix([0,x,y,z])
qmodel = sqrt(q1**2+q2**2+q3**2+q4**2)
Rotation_Matrix = Matrix([[1,0,0,0],[0,q1**2+q2**2-q3**2-q4**2, 2*q2*q3-2*q1*q4, 2*q2*q4+2*q1*q3],[0,2*q2*q3+2*q1*q4,q1**2-q2**2+q3**2-q4**2,2*q3*q4-2*q1*q2],[0,2*q2*q4-2*q1*q3,2*q3*q4+2*q1*q2,q1**2-q2**2-q3**2+q4**2]])
Output = Rotation_Matrix*Input_Matix/qmodel
M = Matrix([[1,3,2],[5,6,3]])

#print(M.shape)#打印矩阵的维度
"""删除矩阵的行或列，用 row_del和col_del"""
M.row_del(0)
Output = Output.row_del(0)
print(Output)
print(M)
print(Rotation_Matrix*Input_Matix/qmodel)
print(qmodel)
