import numpy as np

class Real_Location():
    def Projection_Matrix:
        pixel_x = 640
        pixel_y = 480
        span_x = 375  # 280mm
        span_y = 199  # 177mm
        D = 265  # Camera distance z = -265mm
        Wide_Input = np.mat([[0, 0, D / 2, 1], [-(span_x / 2), 0, D / 2, 1], [span_x / 2, 0, D / 2, 1]])
        Projection_Matrix = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1 / D], [0, 0, 0, 1]])
        Pixel_Transform_Matrix = np.mat([[pixel_x / span_x, 0, 0, 0], [0, pixel_y / span_y, 0, 0], [0, 0, 1, 0], [pixel_x / 2, pixel_y / 2, 0, 1]])
        # Pf= np.array([[(1/r)*sympy.cot(a/2),0,0,0],[0,sympy.cot(a/2),0,0],[0,0,f/(f-n),(-f*n)/(f-n)],[0,0,1,0]])    #定义一个数组
        print(Wide_Input * Projection_Matrix)

        Whole_Matrix = Projection_Matrix * Pixel_Transform_Matrix
        print(Pixel_Transform_Matrix)
        print(Whole_Matrix)
        Input = np.mat([0, 0, 0, 1])
        print(Input * Whole_Matrix)