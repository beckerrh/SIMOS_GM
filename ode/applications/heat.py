from SIMOS_GM.ode import classes
import numpy as np
import scipy.sparse as sparse

#------------------------------------------------------------------
class Heat(classes.Application):
    def __init__(self, n=101):
        h = 1/(n-1)
        self.dim = n
        dataA = [-np.ones(n)/h, 2 * np.ones(n)/h, -np.ones(n)/h]
        dataM = [h*np.ones(n)/6, 2*h * np.ones(n)/3, h*np.ones(n)/6]
        offsets = np.array([-1, 0, 1])
        self.A = sparse.dia_matrix((dataA, offsets), shape=(n, n))
        self.M = sparse.dia_matrix((dataM, offsets), shape=(n, n))
        self.A = self.A.todense()
        self.M = self.M.todense()
        u0 = np.zeros(n)
        u0[n//2] = 1
        super().__init__(u0=u0, T=1)
        self.f = lambda u: -self.A@u
        self.df = lambda u: -self.A

# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    linestyle_tuple = [
        # ('loosely dotted', (0, (1, 10))),
        # ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),
        # ('long dash with offset', (5, (10, 3))),
        # ('loosely dashed', (0, (5, 10))),
        # ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        # ('loosely dashdotted', (0, (3, 10, 1, 10))),
        # ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        # ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
    from SIMOS_GM.ode import cgp
    nt = 10
    nx = 101
    app = Heat(n=nx)
    u = {}
    t = np.linspace(0, 0.001, nt)
    x = np.linspace(0, 1, nx)
    n1 = 4*(nx-1)//10
    n2 = 6*(nx-1)//10+1
    for k in range(1,5):
        method = cgp.CgP(k=k, alpha=1)
        sol_ap = method.run_forward(t, app)
        u[k] = method.interpolate(t, sol_ap)
        # if k==1: plt.plot(x[n1:n2], u[1][0][n1:n2], label=f"u0", linewidth=1.75, color='black')
        linestyle = linestyle_tuple[k-1][1]
        plt.plot(x[n1:n2], u[k][-1][n1:n2], linestyle=linestyle, label=f"{k=}")
        # plt.plot(x[n1:n2], u[k][-1][n1:n2], label=f"{k=}", linestyle=linestyle, linewidth=1.75, color='black')
    # plt.plot(x[n1:n2], u[nt//2][n1:n2], label=f"t={t[nt//2]}")
    plt.legend()
    plt.grid()
    plt.show()
