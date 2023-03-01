import sys, os
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import matplotlib.pyplot as plt
from SIMOS_GM.ode import cgp, applications, classes, analytical_solutions
from SIMOS_GM import utility
from SIMOS_GM.ode import classes
from SIMOS_GM.ode import utils
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------
class ParameterTest():
    def __init__(self, method, app, functionals, observations, n0=10, printinfo=False):
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
        self.iter = 0
    def construct_app_d(self, app, i):
        class app_d(classes.Application):
            def __init__(self, app):
                super(app_d, self).__init__(dim=app.dim)
                self.u = None
                self.df = app.df
                self.f = eval(f"app.f_p{i:1d}")
                self.l = eval(f"app.l_p{i:1d}")
                self.u_zero = eval(f"app.u_zero_p{i:1d}")
                self.app = app
        return app_d(app)
    def solve(self, p):
        method, app, functionals, observations, t = self.method, self.app, self.functionals, self.observations, self.t
        app.setParameter(p)
        u_ap = method.run_forward(t, app)
        return method.interpolate(t, u_ap)[0]
    def residuals(self, p):
        method, app, functionals, observations, t = self.method, self.app, self.functionals, self.observations, self.t
        app.setParameter(p)
        u_ap = method.run_forward(t, app)
        self.u_ap = u_ap
        self.est_u = method.estimator(t, u_ap, app)
        self.R.fill(0)
        for i in range(self.n_obs):
            functional, observation = functionals[i], observations[i]
            J_ap = method.compute_functional(t, u_ap, functional)
            self.R[i] = J_ap - observation
        return self.R
    def dresiduals(self, p, stopatfuncerror=True, plotiteration=True, theta=0.8, lambda_ref=0.025):
        method, app, functionals, observations, t = self.method, self.app, self.functionals, self.observations, self.t
        R = self.residuals(p)
        assert len(R)==self.n_obs
        dR = np.empty(shape=(len(R), self.dim))
        z_ap, est_z = [], []
        eta_z = np.full_like(self.est_u[0]['nl'], -np.inf)
        for i in range(self.n_obs):
            functional = functionals[i]
            z_ap.append(method.run_backward(t, self.u_ap, self.app, functional))
            est_z.append(method.estimator_dual(t, self.u_ap, z_ap[-1], app, functional))
            eta_z = np.maximum(eta_z, est_z[-1][0]['nl'] + est_z[-1][0]['ap'])
        eta_p = self.est_u[0]['nl'] + self.est_u[0]['ap']
        est_du, du_aps = [], []
        eta_du = np.full_like(self.est_u[0]['nl'], -np.inf)
        for j in range(self.dim):
            self.app_d[j].setParameter(p)
            du_ap = method.run_forward(t, self.app_d[j], linearization=self.u_ap)
            du_aps.append(du_ap)
            est_du.append(method.estimator(t, du_ap, self.app_d[j]))
            eta_du = np.maximum(eta_du, est_du[-1][0]['nl'] + est_du[-1][0]['ap'])
            for i in range(self.n_obs):
                functional = functionals[i]
                # z_ap = method.run_backward(t, self.u_ap, self.app, functional)
                dZ_ap = method.compute_functional_dual(t, z_ap[i], self.app_d[j], linearization=self.u_ap)
                dJ_ap = method.compute_functional(t, du_ap, functional)
                dR[i,j] = dJ_ap
                if not np.allclose(dJ_ap, dZ_ap):
                    if stopatfuncerror:
                        plt.show()
                        plt.title(f"{p=}")
                        plt.plot(t, method.interpolate(t, self.u_ap)[0], label='u')
                        plt.plot(t, method.interpolate(t, du_ap)[0], label='du')
                        plt.plot(t, method.interpolate_dual(t, z_ap[i])[0], label='dz')
                        plt.grid()
                        plt.legend()
                        plt.show()
                        raise ValueError(f"{abs(dJ_ap-dZ_ap)=:10.3e} {dJ_ap=} {dZ_ap=} {functional.name=}")
                    print(f"{abs(dJ_ap-dZ_ap)=:10.3e} {dJ_ap=} {dZ_ap=} {functional.name=}")
        J2 = np.sum(R**2)
        G = np.sqrt(np.sum((dR.T@R)**2))
        eta_p += np.sqrt(J2)*eta_du
        eta = (np.sum(eta_p) * eta_z + np.sum(eta_z) * eta_p) / (np.sum(eta_z) + np.sum(eta_p))
        # print(f"{eta.shape=} {self.t.shape=}")
        etaval = np.sum(eta)
        # print(f"{np.sqrt(np.sum(eta_p))=} {np.sqrt(np.sum(eta_z))=} {np.sqrt(np.sum(eta_du))=}")
        msg = f"{self.iter:3d} {0.5*J2:11.5e} {G:10.4e} {etaval:10.4e}"
        if etaval > lambda_ref*G:
            msg += "  *"
            self.t, refinfo = utils.adapt_mesh(t, eta, theta=theta)
            # self.dresiduals(p)
        if self.iter==1:
            print(f"{'it':3s} {'J':^11s} {'G':^10s} {'eta':10s} {'ref':3s}")
            print(41*"=")
        print(msg)
        self.iter += 1
        if plotiteration and self.iter%3==0:
            from matplotlib import ticker
            tm = 0.5 * (t[1:] + t[:-1])
            dt = t[1:] - t[:-1]
            plt.show()
            fig = plt.figure(figsize=1.5*plt.figaspect(1))
            figshape = (3, self.n_obs)
            # fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
            fig.suptitle(f"iter={self.iter} n={len(t)}")
            # ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
            # ax1 = fig.add_subplot(*figshape, 1)
            # ax1 = app.plot(fig, t, method.interpolate(t, self.u_ap)[0], axkey=(*figshape, 1), label_ad="u")
            # ax1.plot(t, method.interpolate(t, self.u_ap)[0], label=f"u")
            ax1 = app.plot(fig=fig, t=t, u=method.interpolate(t, self.u_ap)[0], axkey=(*figshape, 1))
            ax1.set_title("u")
            # ax1.legend()
            ax1.grid()
            # ax2 = fig.add_subplot(*figshape, 2)
            for i in range(self.n_obs):
                z_node = method.interpolate_dual(t, z_ap[i])[0]
                ax2 = app.plot(fig=fig, t=t, u=z_node, axkey=(*figshape, 4+i))
                # app.plotax(t, z_node, ax=ax2)
                # for k in range(z_node.shape[1]):
                #     app.plotax(t, z_node[:,k], label=f"z_{i}({k})", ax=ax2)
                    # ax2.plot(t, z_node[:, k], label=f"z_{i}({k})")
                ax2.set_title(f"z_{i}")
                ax2.grid()
            # ax3 = fig.add_subplot(*figshape, 3, sharex = ax1)
            for j in range(self.dim):
                du_node = method.interpolate(t, du_aps[j])[0]
                ax3 = app.plot(fig=fig, t=t, u=du_node, axkey=(*figshape, 7+j))
                # for k in range(du_node.shape[1]):
                #     ax3.plot(t, du_node[:, k], label=f"du_{j}({k})")
            # ax3.legend()
                ax3.set_title(f"du_{j}")
                ax3.grid()
            # plt.plot(tm, eta, label=f"eta")
            # plt.legend()
            # plt.grid()
            ax2 = fig.add_subplot(*figshape, 2)
            # ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
            ax2.plot(tm, dt, '-b', label=r'$\delta$')
            ax2.legend()
            ax3.grid
            ax3 = fig.add_subplot(*figshape, 3)
            # ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
            ax3.plot(tm, eta, '-g', label=r'$\eta$')
            ax3.legend()
            ax3.grid
            plt.show()
        return dR
    def f(self, p):
        R = self.residuals(p)
        return 0.5*np.sum(R**2)
    def grad(self, p):
        dR = self.dresiduals(p)
        return dR.T@self.R

#------------------------------------------------------------------
def optimize(x0, method, app, functionals, observations, n0, plot=False):
    pt = ParameterTest(method=method, app=app, functionals=functionals, observations=observations, n0=n0, printinfo=True)
    from scipy.optimize import minimize, leastsq, least_squares
    from SIMOS_GM import gm, utility
    # method = 'minimize'
    method = 'leastsq'
    # method = 'least_squares'
    # method = 'gm'
    if plot:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.show()
    if method=='gm':
        datain = utility.algodata.AlgoInput(history=False)
        datain.t = 0.000001
        datain.mu = 0.1
        datain.maxiter = 10000
        xs = gm.gm(np.zeros(pt.dim), pt.f, pt.grad, datain=datain)
    elif method == 'minimize':
        res = minimize(fun= pt.f, x0=x0, jac=pt.grad, tol=1e-12)
        # res = minimize(fun= pt.f, x0=x0)
        xs = res["x"]
        print(res)
    elif method == 'leastsq':
        res = leastsq(pt.residuals, x0=x0, Dfun=pt.dresiduals, ftol=1e-10, xtol=1e-14, gtol=0, full_output=True)
        xs = res[0]
        info = res[2]
        success = res[4]
        print(f"{info['nfev']=} {info['njev']=} {success=}")
    elif method == 'least_squares':
        res = least_squares(pt.residuals, x0=x0, jac=pt.dresiduals, ftol=1e-14, xtol=1e-14, gtol=1e-12, method='lm')
        xs = res.x
        print(res)
    print(f"{x0=} {xs=}")
    if plot:
        u0 = pt.solve(x0)
        us = pt.solve(xs)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        figshape = (1, 2)
        ax = app.plot(fig, pt.t, u0, axkey=(*figshape, 1), label_ad="u0")
        ax.legend()
        ax =app.plot(fig, pt.t, us, axkey=(*figshape, 2), label_ad="us")
        ax.legend()
        plt.show()

#------------------------------------------------------------------
def test_gradient(method, app, functionals, observations, x, n0, plot=False):
    pt = ParameterTest(method=method, app=app, functionals=functionals, observations=observations, n0=n0)
    fig = plt.figure()
    if pt.dim==1 and plot:
        nx = len(x)
        eps = (x[-1]-x[0])/10
        plt.plot(x, [pt.f(xi) for xi in x], label="J(p)")
        for i in [nx//3+i*nx//6 for i in range(3)]:
            xi = x[i]
            f = pt.f(xi)
            g = pt.grad(xi)
            # print(f"{i=} {xi=} {f=} {g=} {(pt.f(xi+1e-4)-pt.f(xi-1e-4))/1e-4/2}")
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
    example = 'lorenz'
    # example = 'bock'
    if example == 'bock':
        F = [classes.FunctionalEndTime(0), classes.FunctionalMean(0)]
        C = np.array([0,2/np.pi])
        x = np.linspace(-np.pi,3*np.pi, 50)
        app = applications.BockTest(mu2=0)
        n0 = 3
        x0 = 1
    elif example == 'exp':
        F = [classes.FunctionalEndTime(), classes.FunctionalMean()]
        app = analytical_solutions.Exponential(T=2,u0=10)
        x = np.linspace(-0.1,0.9, 5), np.linspace(-1,14, 5)
        C = np.array([16,20])
        n0 = 10
    elif example == 'sin':
        F = [classes.FunctionalEndTime(0), classes.FunctionalEndTime(1), classes.FunctionalMean()]
        app = analytical_solutions.SinusIntegration(T=4)
        x = np.linspace(-1,4, 5), np.linspace(-1,4, 5)
        C = np.array([1,0,0])
        n0 = 10
    elif example == 'osc':
        F = [classes.FunctionalEndTime(0), classes.FunctionalEndTime(1)]
        # F = [classes.FunctionalEndTime(1)]
        app = analytical_solutions.Oscillator()
        x = np.linspace(0.03, 1.4, 60)
        C = np.array([1,-1])
        # C = np.array([1])
        x0 = np.array([0.1])
        n0 = 10
    elif example == 'lorenz':
        F = [classes.FunctionalEndTime(0), classes.FunctionalEndTime(1), classes.FunctionalEndTime(2)]
        app = applications.Lorenz(T=15, param=[0,1,2])
        #sigma=10, rho=28, beta=8/3
        #u_node[-1]=array([-10.30698237,  -4.45105613,  35.09463786]) Fmean(0) = -1.13983673e+02
        x0 = np.array([10, 28, 8/3])
        # x0 = np.array([10, 28])
        # x0 = np.array([28])
        x = np.linspace(8, 12, 10)
        C = np.array([-10.30698237,  -4.45105613,  35.09463786, -100])
        #tough
        C = np.array([10.,  -10.,  30.])
        C = np.array([20.,  10.,  30.])
        # C = np.array([10.,  30.,  10.])
        #rho
        x = np.linspace(20, 30, 40)
        #sigma
        x = np.linspace(8.9, 12.5, 40)
        x = np.linspace(9.2, 11.5, 40),np.linspace(20, 28, 40)
        n0 = 1000
    else:
        raise ValueError(f"{example=} unknown")
    # test_gradient(method=cgp.CgP(k=3), app=app, functionals=F, observations=C, x=x, n0=n0, plot=True)
    optimize(x0=x0, method=cgp.CgP(k=20), app=app, functionals=F, observations=C, n0=n0, plot=True)