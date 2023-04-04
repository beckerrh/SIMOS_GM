from SIMOS_GM.ode import classes
import numpy as np
#------------------------------------------------------------------
class BockTest(classes.Application):
    def __init__(self, mu2=1):
        self.mu2 = mu2
        self.p = np.pi
        self.nparam = 1
        super().__init__(u0=[0,np.pi], T=1)
        self.f = lambda u: [u[1], self.mu2*u[0]]
        self.df = lambda u: [[0,1],[self.mu2,0]]
        self.l = lambda t: np.array([0*t, -(self.mu2+self.p**2)*np.sin(self.p*t)])
        self.sol_ex = lambda t: [np.sin(self.p*t), self.p*np.cos(self.p*t)]
        self.dsol_ex = lambda t: np.array([self.p*np.cos(self.p*t), -self.p**2*np.sin(self.p*t)])
        self.f_p0 = lambda u: [0*u[0],0*u[0]]
        # self.l_p0 = lambda t: np.array([0*t, -2*self.p*np.sin(self.p*t)- (self.mu2+self.p**2)*t*np.cos(self.p*t)])
        self.l_p0 = lambda t: [0*t, -2*self.p*np.sin(self.p*t)- (self.mu2+self.p**2)*t*np.cos(self.p*t)]
        self.u_zero_p0 = lambda : [0,0]

    def setParameter(self, p): self.p = p
