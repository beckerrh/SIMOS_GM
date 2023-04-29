from SIMOS_GM.ode import classes
import numpy as np

#------------------------------------------------------------------
class VanDerPol(classes.Application):
    def __init__(self, mu=50):
        super().__init__(u0=[2,0], T=100)
        self.mu = mu
        self.f = lambda u: [u[1], self.mu*(1-u[0]**2)*u[1] - u[0]]
        self.df = lambda u: [[0,1], [-2*self.mu*u[0]*u[1]-1, self.mu*(1-u[0]**2)]]

#------------------------------------------------------------------
if __name__ == "__main__":
    from SIMOS_GM.ode import compare, cgp

    methods = [cgp.CgP(k=1), cgp.CgP(k=2), cgp.CgP(k=3), cgp.CgP(k=4), cgp.CgP(k=5), cgp.CgP(k=6)]
    compare_methods.compare_methods(VanDerPol(mu=0.8), methods, n=100)
