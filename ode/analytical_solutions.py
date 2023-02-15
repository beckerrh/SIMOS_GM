import numpy as np
from classes import Application

#------------------------------------------------------------------
class Exponential(Application):
    def __init__(self, rate=0.2, T=15):
        super().__init__(u0=1, T=15)
        self.f = lambda u: [rate*u]
        self.df = lambda u: [rate]
        self.sol_ex = lambda t: [np.exp(rate*t)]
        self.dsol_ex = lambda t: [rate*np.exp(rate*t)]
#------------------------------------------------------------------
class Oscillator(Application):
    def __init__(self):
        super().__init__(u0=[1, 0], T=4*np.pi)
        self.f = lambda u: [u[1], -u[0]]
        self.df = lambda u: [[0, 1], [-1, 0]]
        self.sol_ex = lambda t: [np.cos(t), -np.sin(t)]
        self.dsol_ex = lambda t: [-np.sin(t), -np.cos(t)]
#------------------------------------------------------------------
class Linear(Application):
    def __init__(self, u0=1, slope=0.2):
        self.slope = slope
        super().__init__(u0=u0, T=5)
        self.f = lambda u: [slope]
        self.df = lambda u: [0]
        self.sol_ex = lambda t: [u0+slope*t]
        self.dsol_ex = lambda t: [slope+0*t]


#------------------------------------------------------------------
class Quadratic(Application):
    def __init__(self, u0=1):
        super().__init__(u0=u0, T=0.9/u0)
        self.f = lambda u: [u**2]
        self.df = lambda u: [2*u]
        self.sol_ex = lambda t: [u0/(1-u0*t)]
        self.dsol_ex = lambda t: [u0**2/(1-u0*t)**2]
#------------------------------------------------------------------
class QuadraticIntegration(Application):
    # u'=f(u)      2t = f(u0+t**2) 2*sqrt(w-u0) = f(w)
    def __init__(self, u0=1):
        super().__init__(u0=u0, T=2)
        self.f = lambda u: [0]
        self.df = lambda u: [0]
        self.sol_ex = lambda t: [u0+t*t]
        self.dsol_ex = lambda t: [2*t]
    def l(self, t):
        return self.dsol_ex(t)
#------------------------------------------------------------------
class LinearIntegration(Application):
    # u'=f(u)      2t = f(u0+t**2) 2*sqrt(w-u0) = f(w)
    def __init__(self, u0=1):
        super().__init__(u0=u0, T=2)
        self.f = lambda u: np.zeros_like(u)
        self.df = lambda u: np.zeros_like(u)
        self.sol_ex = lambda t: [1+t]
        self.dsol_ex = lambda t: [1+0*t]
    def l(self, t):
        return self.dsol_ex(t)
    # def f(self, u):
    #     return np.zeros_like(u)

