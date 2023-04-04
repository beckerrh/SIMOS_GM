from SIMOS_GM.ode import classes
import numpy as np
#------------------------------------------------------------------
class RobertsonAutocatalytic(classes.Application):
    def __init__(self):
        super().__init__(u0=[1,0,0], T=500)
        # xdot = -0.04 * x + 1.e4 * y * z
        # ydot = 0.04 * x - 1.e4 * y * z - 3.e7 * y ** 2
        # zdot = 3.e7 * y ** 2
        self.f = lambda u: [-0.04*u[0] + 1.e4*u[1]*u[2], 0.04*u[0] - 1.e4*u[1]*u[2] - 3.e7*u[1]** 2, 3.e7*u[1]** 2]
        self.df = lambda u: [[-0.04,1.e4*u[2],1.e4*u[1] ], [0.04,-1.e4*u[2]-6.e7*u[1],-1.e4*u[1]], [0, 6.e7*u[1],0]]

