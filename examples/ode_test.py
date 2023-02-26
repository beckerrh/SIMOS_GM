import sys, os
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from SIMOS_GM.ode import cgp, applications, classes, analytical_solutions
from SIMOS_GM import utility
from SIMOS_GM.ode import classes
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------
class ParameterTest():
    def __init__(self, method, app, functionals, observations, n0=10):
        assert len(observations)==len(functionals)
        self.method, self.app, self.functionals, self.observations = method, app, functionals, observations
        self.dim = app.nparam
        self.n_obs = len(observations)
        self.t = np.linspace(0, app.T, n0)
        self.R = np.zeros(self.n_obs)
        self.app_d = []
        for j in range(self.dim):
            self.app_d.append(self.construct_app_d(app,j))
            # print(f"{self.app_d[-1].name=}")
            self.app_d[-1].u_zero()
    def construct_app_d(self, app, i):
        from functools import partial
        class app_d(classes.Application):
            def __init__(self, app):
                super(app_d, self).__init__(dim=app.dim)
                self.u = None
                self.df = app.df
                self.f = eval(f"app.f_p{i:1d}")
                self.l = eval(f"app.l_p{i:1d}")
                self.u_zero = eval(f"app.u_zero_p{i:1d}")
                # self.f = partial(app.f_p, j=i)
                # self.l = partial(app.l_p, j=i)
                # self.u_zero = partial(app.u_zero_p, j=i)
                self.app = app
        return app_d(app)
    def f(self, p):
        method, app, functionals, observations, t = self.method, self.app, self.functionals, self.observations, self.t
        app.setParameter(p)
        u_ap = method.run_forward(t, app)
        self.u_ap = u_ap
        est, estval = method.estimator(t, u_ap, app)
        self.R.fill(0)
        for i in range(self.n_obs):
            functional, observation = functionals[i], observations[i]
            J_ap = method.compute_functional(t, u_ap, functional)
            self.R[i] = J_ap-observation
        return 0.5*np.sum(self.R**2)
    def grad(self, p, stopatfuncerror=True):
        method, app, functionals, observations, t = self.method, self.app, self.functionals, self.observations, self.t
        self.f(p)
        # fig, axs = plt.subplots(nrows=self.dim+1, ncols=1, sharex=True)
        # fig.suptitle(f"{method.name} {app.name} (p={p})")
        u_node, u_mid = method.interpolate(t, self.u_ap)
        # axs[0].plot(self.t, u_node, label='u')
        # axs[1].plot(self.t, self.t * u_node[:, 0], '--', label=f't*u')
        g = np.zeros(self.dim)
        for j in range(self.dim):
            self.app_d[j].setParameter(p)
            du_ap = method.run_forward(t, self.app_d[j], linearization=self.u_ap)
            du_node, du_mid = method.interpolate(t, du_ap)
            # axs[j+1].plot(self.t, du_node, '-', label=f'du_{j:1d}')
            for i in range(self.n_obs):
                functional = functionals[i]
                z_ap = method.run_backward(t, self.u_ap, self.app, functional)
                dZ_ap = method.compute_functional_dual(t, z_ap, self.app_d[j], linearization=self.u_ap)
                dJ_ap = method.compute_functional(t, du_ap, functional)
                if not np.allclose(dJ_ap, dZ_ap):
                    if stopatfuncerror:
                        plt.show()
                        plt.title(f"{p=}")
                        plt.plot(t, u_node, label='u')
                        plt.plot(t, method.interpolate(t, du_ap)[0], label='du')
                        plt.plot(t, method.interpolate_dual(t, z_ap)[0], label='dz')
                        plt.grid()
                        plt.legend()
                        plt.show()
                        raise ValueError(f"{dJ_ap=} {dZ_ap=} {functional.name=}")
                    print(f"{dJ_ap=} {dZ_ap=} {functional.name=}")
                # print(f"uT={t[-1]*u_node[-1]} {dJ_ap=} {t[-1]*u_node[-1]/dJ_ap=}")
                g[j] += self.R[i]*dJ_ap
        # for ax in axs: ax.legend(); ax.grid()
        # plt.show()
        return g



#------------------------------------------------------------------
def optimize(method, app, functionals, observations):
    pt = ParameterTest(method=method, app=app, functionals=functionals, observations=observations)
    from scipy.optimize import minimize, leastsq
    from SIMOS_GM import gm, utility
    datain = utility.algodata.AlgoInput()
    datain.history = False
    datain.t = 0.0001
    datain.mu = 0.1
    datain.maxiter = 10000
    # xsol = gm.gm(np.zeros(pt.dim), pt.f, pt.grad, datain=datain)
    # print(f"{xsol=}")
    res = minimize(fun= pt.f, x0=np.zeros(pt.dim), jac=pt.grad)
    # res = minimize(fun= pt.f, x0=np.zeros(pt.dim))
    print(res)

#------------------------------------------------------------------
def test_gradient(method, app, functionals, observations, x, plot=False):
    pt = ParameterTest(method=method, app=app, functionals=functionals, observations=observations)
    fig = plt.figure()
    if pt.dim==1 and plot:
        nx = len(x)
        eps = (x[-1]-x[0])/10
        plt.plot(x, [pt.f(xi) for xi in x], label="J(p)")
        for i in [nx//3+i*nx//6 for i in range(3)]:
            xi = x[i]
            f = pt.f(xi)
            g = pt.grad(xi)
            # print(f"{i=} {xi=} {f=} {g=}")
            plt.plot(xi, f, 'xr')
            plt.plot([xi-eps, xi+eps], [f-eps*g, f+eps*g], '--y', label=f"TJ({xi:5.2f})")
    elif pt.dim==2 and plot:
        xm, ym = np.meshgrid(*x)
        zm = np.zeros_like(xm)
        for i in range(xm.shape[0]):
            for j in range(xm.shape[1]):
                zm[i,j] = pt.f(np.array([xm[i,j],ym[i,j]]))
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(xm, ym, zm, cmap=cm.gnuplot_r, alpha=0.7)
        ax.contour(xm, ym, zm, np.linspace(0, 20, 10), offset=-1, linewidths=2, cmap=cm.gray)
        ax.view_init(26, 130)
        plt.draw()
        plt.legend()
    plt.show()
    # xs = np.tile(x,pt.dim).reshape((nx,pt.dim))
    maxerr = utility.derivativetest.test_grad(pt.f, pt.grad, x)
    print(f"{maxerr=}")
#------------------------------------------------------------------
if __name__ == "__main__":
    example = 'osc'
    if example == 'bock':
        F = [classes.FunctionalEndTime(), classes.FunctionalMean()]
        C = np.array([0,1])
        x = np.linspace(-np.pi,3*np.pi, 50)
        app = applications.BockTest(mu2=0)
    elif example == 'exp':
        F = [classes.FunctionalEndTime(), classes.FunctionalMean()]
        app = analytical_solutions.Exponential(T=2,u0=10)
        x = np.linspace(-0.1,0.9, 5), np.linspace(-1,14, 5)
        C = np.array([16,20])
    elif example == 'osc':
        F = [classes.FunctionalEndTime(0), classes.FunctionalEndTime(1), classes.FunctionalMean(0)]
        F = [classes.FunctionalEndTime(1)]
        app = analytical_solutions.Oscillator()
        x = np.linspace(0.1, 0.5, 40)
        C = np.array([2,-1,0])
        C = np.array([1])
    elif example == 'sin':
        F = [classes.FunctionalEndTime(), classes.FunctionalMean()]
        app=analytical_solutions.LinearIntegration()
        x = np.linspace(-1,1, 50)
        C = np.array([0,1])
    else:
        raise ValueError(f"{example=} unknown")
    test_gradient(method=cgp.CgP(k=1), app=app, functionals=F, observations=C, x=x, plot=True)
    # optimize(method=cgp.CgP(k=1), app=app, functionals=F, observations=C)