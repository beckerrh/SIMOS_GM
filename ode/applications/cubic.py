from SIMOS_GM.ode import classes
import numpy as np


#------------------------------------------------------------------
class Cubic(classes.Application):
    def __init__(self, u0=1/100, scale=500):
        super().__init__(u0=u0, T=1)
        self.f = lambda u: [scale*u**2*(1-u)]
        self.df = lambda u: [scale*(2*u-6*u**2)]
