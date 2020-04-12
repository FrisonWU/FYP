import sympy
from sympy import DiracDelta
import numpy as np

sympy.init_printing(use_latex=True)
x, x_vel, y, y_vel = sympy.symbols('x, x_vel, y, y_vel')
H = np.array([[x,0,0,0],[0,0,0,0],[0,0,y,0],[0,0,0,0]])
state = sympy.Matrix([x, x_vel, y, y_vel])
print(sympy.factor(sympy.simplify(H.jacobian(state))))