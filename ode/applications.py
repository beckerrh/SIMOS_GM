from SIMOS_GM.ode import classes
import numpy as np


#------------------------------------------------------------------
class Cubic(classes.Application):
    def __init__(self, u0=1/100, scale=500):
        super().__init__(u0=u0, T=1)
        self.f = lambda u: [scale*u**2*(1-u)]
        self.df = lambda u: [scale*(2*u-6*u**2)]

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

#------------------------------------------------------------------
class Lorenz(classes.Application):
    def __init__(self, T=30, sigma=10, rho=28, beta=8/3, param=None):
        # super().__init__(u0=[-10, -4.45, 35.1], T=20)
        super().__init__(u0=[1,0,0], T=T)
        # print(f"{param=}")
        if param is not None:
            self.nparam = len(param)
            self.param = param
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: [self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]]
        # self.df = lambda u: np.array([[-self.sigma, self.sigma,0], [rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]])
        self.df = lambda u: [[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]]
        self.f_p0 = lambda u: [0 * u[0], 0 * u[1], 0 * u[2]]
        self.f_p1 = lambda u: [0 * u[0], 0 * u[1], 0 * u[2]]
        self.f_p2 = lambda u: [0 * u[0], 0 * u[1], 0 * u[2]]
        self.l_p0 = lambda t: [0*t,0*t,0*t]
        self.l_p1 = lambda t: [0*t,0*t,0*t]
        self.l_p2 = lambda t: [0*t,0*t,0*t]
        if param is not None:
            for i in range(len(param)):
                if self.param[i]==0:
                    exec(f"self.f_p{i:1d} = lambda u: [u[1]-u[0], 0*u[1], 0*u[2]]")
                if self.param[i]==1:
                    exec(f"self.f_p{i:1d} = lambda u: [0*u[0], u[0], 0*u[2]]")
                if self.param[i]==2:
                    exec(f"self.f_p{i:1d} = lambda u: [0*u[0], 0*u[1], -u[2]]")
        self.u_zero_p0 = lambda: [0, 0, 0]
        self.u_zero_p1 = lambda: [0, 0, 0]
        self.u_zero_p2 = lambda: [0, 0, 0]
    def setParameter(self, p):
        # self.sigma, self.rho, self.beta = p[0], p[1], p[2]
        p = np.atleast_1d(p)
        if not len(p)==len(self.param):
            raise ValueError(f"{p=} {self.param=}")
        for i in range(len(p)):
            if self.param[i]==0: self.sigma=p[i]
            if self.param[i]==1: self.rho=p[i]
            if self.param[i]==2: self.beta=p[i]
        # print(f"{p=} {self.sigma=}  {self.rho=} {self.beta=}")
    def plotax(self, t, u, ax, label=""):
        import matplotlib.pyplot as plt
        x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label=label)
        ax.plot(x[-1], y[-1], z[-1], 'Xr', label=label+"(T)")
        ax.plot(x[0], y[0], z[0], 'Xy', label=label+"(0)")
        # ax.legend()
        # ax.plot_surface(x, y, z, cmap=cm.gnuplot_r, alpha=0.7)
        # ax.contour(x, y, z, np.linspace(0,20,10), offset=-1, linewidths=2, cmap=cm.gray)
        ax.view_init(26, 130)
        plt.draw()
        return ax

    def plot(self, fig, t, u, axkey=(1,1,1), label_u=r'$u_{\delta}$', label_ad=''):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        # print(f"{axkey=}")
        ax = fig.add_subplot(*axkey, projection='3d')
        return self.plotax(t, u, ax, label_u+label_ad)


