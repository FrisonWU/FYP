from sympy import *
import numpy as np
pixel_x = 640
pixel_y = 480
span_x = 375  # 280mm
span_y = 199  # 177mm
D = 265  # Camera distance z = -475mm
Projection_Matrix = Matrix([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 1 / D],
                              [0, 0, -D, 1]])
Pixel_Transform_Matrix = Matrix([[pixel_x / span_x, 0, 0, 0],
                                   [0, pixel_y / span_y, 0, 0],
                                   [0, 0, 1, 0],
                                   [pixel_x / 2, pixel_y / 2, 0, 1]])
Project_Pixel = Projection_Matrix * Pixel_Transform_Matrix
wx, wy, wz, q1, q2, q3, q4 = symbols('wx, wy, wz, q1, q2, q3, q4')
Input_Matrix = Matrix([0, wx, wy, wz])
qmodel = sqrt(q1 ** 2 + q2 ** 2 + q3 ** 2 + q4 ** 2)
Rotation_Matrix = Matrix([[1, 0, 0, 0],
                          [0, q1 ** 2 + q2 ** 2 - q3 ** 2 - q4 ** 2, 2 * q2 * q3 - 2 * q1 * q4,
                           2 * q2 * q4 + 2 * q1 * q3],
                          [0, 2 * q2 * q3 + 2 * q1 * q4, q1 ** 2 - q2 ** 2 + q3 ** 2 - q4 ** 2,
                           2 * q3 * q4 - 2 * q1 * q2],
                          [0, 2 * q2 * q4 - 2 * q1 * q3, 2 * q3 * q4 + 2 * q1 * q2,
                           q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2]])
Output_Matrix = Rotation_Matrix * Input_Matrix / qmodel
wx_, wy_, wz_ = Output_Matrix.row(1), Output_Matrix.row(2), Output_Matrix.row(3)
Rotated_World = Matrix([wx_, wy_, wz, 1])
Rotated_Pixel = Project_Pixel * Rotated_World
Rotated_Pixel_Homo = Rotated_Pixel / Rotated_Pixel.row(3)
calx = Rotated_Pixel_Homo.row(0)
caly = Rotated_Pixel_Homo.row(1)
state = Matrix([wx, wy, wz, q1, q2, q3, q4])
X_Jaco = calx.jacobian(state)
Y_Jaco = caly.jacobian(state)
X_Jaco=X_Jaco.subs(wx,2).subs(wy,2).subs(wz,2).subs(q1,2).subs(q2,2).subs(q3,2).subs(q4,2)
Y_Jaco=Y_Jaco.subs(wx,2).subs(wy,2).subs(wz,2).subs(q1,2).subs(q2,2).subs(q3,2).subs(q4,2)
#Y_Jaco.sub(wx,2).sub(wy,2).sub(wz,2).sub(q1,2).sub(q2,2).sub(q3,2).sub(q4,2)
X_NUM = np.array(X_Jaco)
Y_NUM = np.array(Y_Jaco)
print(X_Jaco)
a = np.array([1,0,0,0])
print(a.reshape(a.shape[0],1))
#print(X_NUM)
tot=np.vstack((X_NUM,Y_NUM))
print(tot)