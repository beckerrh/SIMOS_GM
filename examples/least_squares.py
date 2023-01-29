import numpy as np

class LeastSquares():
    """
    see scipy.optimize.rosen
    """
    def __init__(self, dim, m, mu=0.1):
        self.dim, self.m, self.mu = dim, m, mu
        self.A = np.random.rand(m, dim)
        self.b = np.random.rand(m)
        from scipy.sparse.linalg import svds
        smin = svds(self.A, k=1, which='SM',return_singular_vectors=False)
        smax = svds(self.A, k=1, which='LM',return_singular_vectors=False)
        self.mu, self.L = float(smin**2), float(smax**2)

    def f(self, x):
        x = np.asarray(x)
        if self.dim!=x.shape[0]: raise ValueError(f"{self.dim=} {x.shape=}")
        r = self.A@x - self.b
        return 0.5*self.mu*np.dot(x,x) + 0.5*np.dot(r,r)
    def grad(self, x):
        if self.dim!=x.shape[0]: raise ValueError(f"{self.dim=} {x.shape=}")
        x = np.asarray(x)
        r = self.A@x - self.b
        g = self.mu* x + self.A.T@r
        return g


def plot_surface():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    def plotter(elev, azim):
        ax = plt.subplot(111, projection='3d')
        x = np.linspace(-20,20)
        y = np.linspace(-20,20)
        x,y = np.meshgrid(x,y)
        ls = LeastSquares(dim=2, m=4)
        z = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i,j] = ls.f([x[i,j],y[i,j]])
        ax.plot_surface(x, y, z, cmap=cm.gnuplot_r)
        ax.contour(x, y, z, offset=-1, cmap=cm.gray)
        ax.view_init(elev=elev, azim=azim)
        plt.draw()
        plt.show()
    plotter(40,10)
    # from ipywidgets import interactive
    # iplot = interactive(plotter, elev=(-90,90,5), azim=(-90,90,5))
    # iplot

def grad_test():
    import sys, os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from SIMOS_GM import utility
    dim=4
    ls = LeastSquares(dim=dim, m=2*dim)
    x = np.linspace(-2,2, 10)
    xs = np.tile(x,dim).reshape((10,4))
    maxerr = utility.derivativetest.test_grad(ls.f, ls.grad, xs)
    print(f"{maxerr=}")

if __name__ == "__main__":
    plot_surface()
    # grad_test()
