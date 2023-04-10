if __name__ == "__main__":
    import sys, os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_DIR =  os.path.abspath(os.path.join(__file__ ,"../../../.."))
    print(f"{SCRIPT_DIR=}")
    sys.path.append(SCRIPT_DIR)

from SIMOS_GM.ode import classes
# from .. import classes
# import classes
import numpy as np

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

    def plot(self, fig, t, u, axkey=(1,1,1), label_u=r'$u_{\delta}$', label_ad='', title=''):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        # print(f"{axkey=}")
        ax = fig.add_subplot(*axkey, projection='3d')
        if title: ax.set_title(title)
        return self.plotax(t, u, ax, label_u+label_ad)


def random(n=1000, niter=1000, nplots=3):
    import matplotlib.pyplot as plt
    from SIMOS_GM.ode import cgp
    fig = plt.figure(figsize=plt.figaspect(1/3))
    app = Lorenz()
    method = cgp.CgP(k=2)
    t = np.linspace(0, app.T, n)
    uplots = []
    for iter in range(niter):
        sol_ap = method.run_forward(t, app, random=True)
        u_ap, u_apm = method.interpolate(t, sol_ap)
        if not iter: u_ap_mean = np.zeros_like(u_ap)
        u_ap_mean += u_ap/niter
        if iter//nplots == 0: uplots.append(u_ap)
    nplots = len(uplots)
    for i in range(nplots): 
        app.plot(fig, t, uplots[i], axkey=(1, nplots+1, i+1), title=f"u{i}")
    app.plot(fig, t, u_ap_mean, axkey=(1, nplots+1, nplots+1), title="umean")
    plt.show()

def compare():
    from SIMOS_GM.ode import compare_methods, cgp
    methods = [cgp.CgP(k=1), cgp.CgP(k=2), cgp.CgP(k=3)]
    compare_methods.compare_methods(Lorenz(), methods, n=1000)

def adaptive():
    from SIMOS_GM.ode import cgp, adaptive
    # X1(30) ≃ −3.892637
    F = classes.FunctionalEndTime(0)
    F = classes.FunctionalMean(0)
    sigma, rho, beta = 41.13548392, 21.54881541,  0.22705707
    app = Lorenz(T=15, sigma=sigma, rho=rho, beta=beta)
    adaptive.run_adaptive(app, cgp.CgP(k=2), n0=1000, itermax=60, F=F, theta=0.9, nplots=20, eps=1e-8)

#------------------------------------------------------------------
if __name__ == "__main__":
    random()


