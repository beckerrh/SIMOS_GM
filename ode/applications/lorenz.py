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
        super().__init__(u0=u0, T=T, nplots=1)
        # print(f"{param=}")
        if param is not None:
            self.nparam = len(param)
            self.param = param
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: [self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]]
        self.df = lambda u: [[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]]
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
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
        # inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs, hspace=0.3)
        # self.plot_histogram(fig, inner[:-1], t, u, label, title)
        # ax = fig.add_subplot(inner[-1], projection='3d')
        ax = fig.add_subplot(gs, projection='3d')
        x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label=label)
        ax.plot(x[-1], y[-1], z[-1], 'Xr', label=label+"(T)")
        ax.plot(x[0], y[0], z[0], 'Xy', label=label+"(0)")
        ax.view_init(26, 130)
        ax.set_title(title)
        plt.draw()
        return ax
    def plot(self, t, u):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label='u', lw=0.5)
        ax.plot(x[-1], y[-1], z[-1], 'X', label="u(T)")
        ax.plot(x[0], y[0], z[0], 'X', label="u(0)")
        ax.plot(*self.FP1, color='k', marker="8", ls='')
        ax.plot(*self.FP2, color='k', marker="8", ls='')
        ax.view_init(26, 130)
        ax.legend()
        plt.show()


#------------------------------------------------------------------
class Lorenz2(Lorenz):
    def __init__(self, T=30, sigma=10, rho=28, beta=8/3, u0=[4, 5, 12], param=None):
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
def read(M=10, q = 0.1, rho=28, T=200, k=1, alpha=0):
    import pathlib
    datadir = pathlib.Path.home().joinpath( 'data_dir' )
    dirs = datadir.glob(f"{q}@{rho}@{T}@*@{M}@{k}@{alpha}")
    print(f"{dirs=}")
    def k(dd):
        return int(str(dd).split('@')[3])
    dirs = sorted(dirs, key=k)
    Hdiffs, udiffs = [], []
    for iter,dir in enumerate(dirs):
        n = int(T * 1e3 * 2 ** iter)
        t = np.linspace(0, T, n)
        if not dir==dirs[0]:
            Hmean_old = Hmean
            u_old = u
        Hmean = np.load(datadir/dir / "H_mean.npy")
        u = np.load(datadir/dir / "u_mean.npy")
        print(f"{np.mean(u,axis=0)=}")
        print(f"{np.max(u,axis=0)=}")
        print(f"{np.min(u,axis=0)=}")
        # Hmean /= np.sum(Hmean)
        if not dir==dirs[0]:
            Hdiffs.append(np.linalg.norm(Hmean-Hmean_old))
            udiff = u[::2]-u_old
            udiffs.append(np.linalg.norm(udiff)/np.sqrt(u_old.shape[0]))
        if iter%4==1:
            for k in range(3):
                plt.subplot(4, 1, k+1)
                plt.plot(t[::2], udiff[:,k], label=f"{iter}_{k}")
                plt.legend()
        # print(f"{dir=} {np.linalg.norm(Hmean)=} {u.shape=}")
    plt.subplot(4, 1, 4)
    print(f"{Hdiffs=} {udiffs=}")
    plt.plot(Hdiffs, 'X-')
    plt.show()
    app = Lorenz()
    app.plot(t=t, u=u, title=f"umean")
    plt.show()


#------------------------------------------------------------------
def random(iter=0, M=1000, q = 0.1, rho=28, T=200, k=1, alpha=0, bins=100):
    # dt = 1e-3 * 2**(iter)
    import matplotlib.pyplot as plt
    from SIMOS_GM.ode import cgp
    import shutil
    import pathlib
    datadir = pathlib.Path.home().joinpath( 'data_dir', f"{q}@{rho}@{T}@{iter}@{M}@{k}@{alpha}")
    try:
        shutil.rmtree(datadir)
    except:
        pass
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)
    app = Lorenz2(rho=rho, T=T)
    method = cgp.CgP(k=k, alpha=alpha)
    n = int(T*1e3*2**iter)
    print(f"{n=} {M=}")
    t = np.linspace(0, app.T, n)
    dt = t[1:] - t[:-1]
    uplots, isamples = [], []
    nplots = 3
    for isample in range(M):
        lintrandom = q * np.sqrt(dt[:,np.newaxis])*np.random.randn(len(dt),3)
        lintrandom -= lintrandom.mean()
        sol_ap = method.run_forward(t, app, lintrandom=lintrandom, q=q)
        u = method.interpolate(t, sol_ap)
        app.change_solution(u)
        if not isample: u_mean = np.zeros_like(u)
        u_mean += u / M
        if (isample*nplots)//M == 0:
            uplots.append(u)
            isamples.append(isample)
        H, edges = np.histogramdd(u, bins=bins, range=[(-20,20), (-25,25), (0,50)], density=True)
        # Hxy, exy1, exy2 = np.histogram2d(u[:,0], u[:,1], bins=(100,100), range=[(-20,20), (-20,20)])
        # Hxz, exz1, exz2 = np.histogram2d(u[:,0], u[:,2], bins=(100,100), range=[(-20,20), (0,50)])
        # Hyz, eyz1, eyz2 = np.histogram2d(u[:,1], u[:,2], bins=(100,100), range=[(-20,20), (0,50)])
        if not isample:
            # Hxy_mean = np.zeros_like(Hxy)
            # Hxz_mean = np.zeros_like(Hxz)
            # Hyz_mean = np.zeros_like(Hyz)
            H_mean = np.zeros_like(H)
        # Hxy_mean += Hxy/M
        # Hxz_mean += Hxz/M
        # Hyz_mean += Hyz/M
        H_mean += H / M
        print(f"{100*(isample / M):4.1f}%")
    # fig = plt.figure(figsize=2*plt.figaspect(1))
    fig = plt.figure()
    plt.suptitle(f"{T=} r={rho} {q=} {n=} nsam={M}")
    outer = gridspec.GridSpec(app.nplots, nplots, wspace=0.2)
    # outer = gridspec.GridSpec( nplots+1, 1, wspace=0.2)
    for i in range(min(nplots,len(uplots))):
        app.plot(fig=fig, gs=outer[0,i], t=t, u=uplots[i], title=f"u{isamples[i]}")
        np.save(datadir / f"uplots_{isamples[i]}", uplots[i])
    plt.show()
    # for l in ["u_mean", "Hxy_mean", "Hxz_mean", "Hyz_mean", "H_mean"]:
    for l in ["u_mean", "H_mean", "t"]:
        np.save(datadir/l, eval(l))
    # app.plot(fig=fig, gs=outer[0,-1], t=t, u=u_mean, title=f"umean")
    cmap = cm.gist_gray_r
    # fig.add_subplot(outer[1,0:2], box_aspect=0.5)
    # plt.title(f"X-Y")
    # X, Y = np.meshgrid(exy1, exy2)
    # plt.pcolormesh(X, Y, Hxy_mean.T, cmap=cmap)
    # fig.add_subplot(outer[1,2:4], box_aspect=0.5)
    # plt.title(f"X-Z")
    # X, Y = np.meshgrid(exz1, exz2)
    # plt.pcolormesh(X, Y, Hxz_mean.T, cmap=cmap)
    # fig.add_subplot(outer[1,4:6], box_aspect=0.5)
    # plt.title(f"Y-Z")
    # X, Y = np.meshgrid(eyz1, eyz2)
    # plt.pcolormesh(X, Y, Hyz_mean.T, cmap=cmap)

    # fig.add_subplot(outer[-1,0:2], box_aspect=0.5)
    # fig.add_subplot(outer[0,0], box_aspect=1)
    plt.subplot(311)
    plt.title(f"X-Y")
    X, Y = np.meshgrid(edges[0], edges[1])
    plt.pcolormesh(X, Y, np.mean(H, axis=2).T, cmap=cmap)
    # fig.add_subplot(outer[-1,2:4], box_aspect=0.5)
    # fig.add_subplot(outer[1,0], box_aspect=1)
    plt.subplot(312)
    plt.title(f"X-Z")
    X, Y = np.meshgrid(edges[0], edges[2])
    plt.pcolormesh(X, Y, np.mean(H, axis=1).T, cmap=cmap)
    # fig.add_subplot(outer[-1,4:], box_aspect=0.5)
    # fig.add_subplot(outer[2,0], box_aspect=1)
    plt.subplot(313)
    plt.title(f"Y-Z")
    X, Y = np.meshgrid(edges[1], edges[2])
    plt.pcolormesh(X, Y, np.mean(H, axis=0).T, cmap=cmap)
    plt.savefig(datadir / "plot.png")
    plt.show()

#------------------------------------------------------------------
def attractor(rho=25, T=1000, dt=1e-3):
    import matplotlib.gridspec as gridspec
    from SIMOS_GM.ode import cgp
    app = Lorenz(T=T, rho=rho)
    method = cgp.CgP(k=1)

    u0s = [[ix,iy,iz] for ix in [-10,10] for iy in [-10,10] for iz in [0,30]]
    eps = 1e-3
    u0s.append([0,0,0 + eps] )
    b = np.sqrt(app.beta*(rho-1))
    u0s.append([b, b,rho-1 + eps])
    u0s.append([-b, -b,rho-1 + eps])
    t = np.arange(0, T, dt)
    print(f"{len(t)=}")
    nrun = int(np.sqrt(len(u0s)-1))+1
    print(f"{nrun=} {len(u0s)=}")
    outer = gridspec.GridSpec(nrun, nrun, wspace=0.2)
    fig = plt.figure()
    for iter,u0 in enumerate(u0s):
        app.u0 = u0
        sol_ap = method.run_forward(t, app)
        u = method.interpolate(t, sol_ap)
        print(f"{np.mean(u, axis=0)=}")
        print(f"{u[-1]=}")
        app.plot(fig=fig, gs=outer[iter], t=t, u=u, title=f"{iter}")
    plt.show()


#------------------------------------------------------------------
if __name__ == "__main__":
    from SIMOS_GM.ode import compare, cgp

    method = cgp.CgP(k=2)
    app = Lorenz()
    t = np.linspace(0, 20, 1000)
    u_node, u_coef = method.run_forward(t, app)
    print(f"{u_node.max()=}")
    app.plot(t, u_node)
