import numpy as np
import scipy.linalg as linalg
import sympy as *

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
        wx_ch, wy_ch, wz_ch = Output_Matrix.row(1), Output_Matrix.row(2), Output_Matrix.row(3)
        Rotated_World = Matrix([wx_ch, wy_ch, wz_ch, 1])
        Rotated_Pixel = Project_Pixel * Rotated_World
        Rotated_Pixel_Homo = Rotated_Pixel / Rotated_Pixel.row(3)