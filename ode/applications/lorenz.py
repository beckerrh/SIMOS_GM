import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

import time

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
    def __init__(self, T=30, sigma=10, rho=28, beta=8/3, u0=[1,0,0], param=None):
        # super().__init__(u0=[-10, -4.45, 35.1], T=20)
        super().__init__(u0=u0, T=T, nplots=4)
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
    def plot_histogram(self, fig, gs, t, u, label="", title=""):
        inner = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=1, subplot_spec=gs, hspace=0.3)
        x,y,z = u[:,0], u[:,1], u[:,2]
        xb = np.histogram_bin_edges(x, bins='auto')
        yb = np.histogram_bin_edges(y, bins='auto')
        zb = np.histogram_bin_edges(z, bins='auto')
        print(f"{len(xb)=} {len(yb)=} {len(zb)=}")
        cmap = cm.gist_gray_r
        axs = [fig.add_subplot(inner[i]) for i in range(3)]
        axs[0].set_title(title)
        axs[0].hist2d(x, y, bins=(xb,yb), cmap=cmap)
        axs[1].hist2d(x, z, bins=(xb,zb), cmap=cmap)
        axs[2].hist2d(y, z, bins=(yb,zb), cmap=cmap)
    def plotax(self, fig, gs, t, u, label="", title=""):
        inner = gridspec.GridSpecFromSubplotSpec(nrows=4, ncols=1, subplot_spec=gs, hspace=0.3)
        self.plot_histogram(fig, inner[:-1], t, u, label, title)
        ax = fig.add_subplot(inner[-1], projection='3d')
        x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label=label)
        ax.plot(x[-1], y[-1], z[-1], 'Xr', label=label+"(T)")
        ax.plot(x[0], y[0], z[0], 'Xy', label=label+"(0)")
        ax.view_init(26, 130)
        plt.draw()
        return ax

#------------------------------------------------------------------
class Lorenz2(Lorenz):
    def __init__(self, T=30, sigma=10, rho=28, beta=8/3, u0=[1,0,0], param=None):
        super().__init__(u0=u0, T=T, sigma=sigma, rho=rho, beta=beta, param=param)
        self.u0[2] -= self.rho+self.sigma
        # print(f"{param=}")
        if param is not None:
            self.nparam = len(param)
            self.param = param
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: [self.sigma*(u[1]-u[0]), -self.sigma*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]]
        self.df = lambda u: [[-self.sigma, self.sigma,0], [-self.sigma-u[2],-1,-u[0]], [u[1],u[0],-self.beta]]
        self.l = lambda t: np.array([np.zeros_like(t),np.zeros_like(t),-self.beta*(self.rho+self.sigma)*np.ones_like(t)])
    def change_solution(self, u): u[:,2] += self.rho+self.sigma


#------------------------------------------------------------------
def compare():
    from SIMOS_GM.ode import compare, cgp
    # compare.compare(Lorenz2(T=100), [cgp.CgP(k=1), cgp.CgP(k=2)], n=10000)
    compare.compare([Lorenz(T=1000), Lorenz2(T=1000)], cgp.CgP(k=1), n=100000)

def adaptive():
    from SIMOS_GM.ode import cgp, adaptive
    # X1(30) ≃ −3.892637
    F = classes.FunctionalEndTime(0)
    F = classes.FunctionalMean(0)
    sigma, rho, beta = 41.13548392, 21.54881541,  0.22705707
    app = Lorenz(T=15, sigma=sigma, rho=rho, beta=beta)
    adaptive.run_adaptive(app, cgp.CgP(k=2), n0=1000, itermax=60, F=F, theta=0.9, nplots=20, eps=1e-8)

#------------------------------------------------------------------
def random(n=1000, nsamples=1000, nplots=3):
    import matplotlib.pyplot as plt
    from SIMOS_GM.ode import cgp
    app = Lorenz2(rho=13)
    fig = plt.figure(figsize=2*plt.figaspect(1))
    outer = gridspec.GridSpec(app.nplots+2, nplots+1, wspace=0.3)
    method = cgp.CgP(k=1)
    t = np.linspace(0, app.T, n)
    dt = t[1:] - t[:-1]
    uplots = []
    for isample in range(nsamples):
        lintrandom = 0.3 * np.sqrt(dt[:,np.newaxis])*np.random.randn(len(dt),3)
        lintrandom -= lintrandom.mean()
        sol_ap = method.run_forward(t, app, lintrandom=lintrandom)
        u = method.interpolate(t, sol_ap)
        app.change_solution(u)
        if not isample: u_mean = np.zeros_like(u)
        u_mean += u/nsamples
        if isample//nplots == 0: uplots.append(u)
        H, edges = np.histogramdd(u, bins=100, range=[(-15,15), (-15,15), (0,20)])
        # H, edges = np.histogramdd(u, bins=100)
        # print(f"{np.linalg.norm(H)=}")
        Hxy, exy1, exy2 = np.histogram2d(u[:,0], u[:,1], bins=(100,100))
        Hxz, exz1, exz2 = np.histogram2d(u[:,0], u[:,2], bins=(100,100))
        Hyz, eyz1, eyz2 = np.histogram2d(u[:,1], u[:,2], bins=(100,100))
        if not isample:
            Hxy_mean = np.zeros_like(Hxy)
            Hxz_mean = np.zeros_like(Hxz)
            Hyz_mean = np.zeros_like(Hyz)
            H_mean = np.zeros_like(H)
        Hxy_mean += Hxy/nsamples
        Hxz_mean += Hxz/nsamples
        Hyz_mean += Hyz/nsamples
        H_mean += H/nsamples
        print(f"{100*(isample/nsamples):4.1f}%")
    nplots = len(uplots)
    for i in range(nplots):
        app.plot(fig=fig, gs=outer[:-2,i], t=t, u=uplots[i], title=f"u{i}")
    app.plot(fig=fig, gs=outer[:-2,-1], t=t, u=u_mean, title=f"umean")
    cmap = cm.gist_gray_r
    fig.add_subplot(outer[-2,0])
    X, Y = np.meshgrid(exy1, exy2)
    plt.pcolormesh(X, Y, Hxy_mean.T, cmap=cmap)
    fig.add_subplot(outer[-2,1])
    X, Y = np.meshgrid(exz1, exz2)
    plt.pcolormesh(X, Y, Hxz_mean.T, cmap=cmap)
    fig.add_subplot(outer[-2,2])
    X, Y = np.meshgrid(eyz1, eyz2)
    plt.pcolormesh(X, Y, Hyz_mean.T, cmap=cmap)

    fig.add_subplot(outer[-1,0])
    X, Y = np.meshgrid(edges[0], edges[1])
    plt.pcolormesh(X, Y, np.mean(H, axis=2), cmap=cmap)
    fig.add_subplot(outer[-1,1])
    X, Y = np.meshgrid(edges[0], edges[2])
    plt.pcolormesh(X, Y, np.mean(H, axis=1), cmap=cmap)
    fig.add_subplot(outer[-1,2])
    X, Y = np.meshgrid(edges[1], edges[2])
    plt.pcolormesh(X, Y, np.mean(H, axis=0), cmap=cmap)

    plt.show()

def attractor():
    import matplotlib.gridspec as gridspec
    from SIMOS_GM.ode import cgp
    # app = Lorenz(T=100, u0=[1,0,0], rho=28)
    T = 100
    rho = 28
    apps = [Lorenz(T=T, u0=[1,0,0], rho=28), Lorenz(T=T, u0=[-10, -4, 35], rho=28)]
    n = 10000
    # t = np.linspace(0, app.T, n)
    methods = [cgp.CgP(k=1), cgp.CgP(k=2)]
    method = cgp.CgP(k=1)
    nsamples = 10
    t = np.linspace(0, T, n)
    dt = t[1:] - t[:-1]
    us={}
    tmean = 0
    for isample in range(nsamples):
        t0 = time.time()
        lintrandom = 10 * np.sqrt(dt[:,np.newaxis])*np.random.randn(len(dt),3)
        lintrandom -= lintrandom.mean()
        for app in apps:
            sol_ap = method.run_forward(t, app, lintrandom=lintrandom)
            u_aps = method.interpolate(t, sol_ap)
            if not isample: us[app] = np.zeros_like(u_aps)
            us[app] += u_aps/nsamples
        if not isample: tmean = 0
        tmean = isample*tmean/nsamples + (nsamples-isample)*(time.time()-t0)/nsamples**2
        perc= isample / nsamples
        total = (1-perc)*tmean*nsamples
        h = int(total/3600)
        m = int((total-3600*h)/60)
        s = total-3600*h -60*m
        print(f"{100*perc:4.1f}% ({h:2d}h {m:2d}m {s:4.1f}s) ")
    pltcount = 0
    fig = plt.figure(figsize=2*plt.figaspect(2))
    fig.suptitle(f"T={T} r={rho}")
    outer = gridspec.GridSpec(2, len(methods), wspace=0.3)
    for app in apps:
        title = f"X0={app.u0}"
        app.plot_histogram(fig=fig, gs=outer[0, pltcount], t=t, u=us[app], title=title)
        app.plot(fig=fig, gs=outer[1, pltcount], t=t, u=us[app], title=title)
        pltcount += 1
    plt.show()


#------------------------------------------------------------------
if __name__ == "__main__":
    # compare()
    random()
    # attractor()


