import numpy as np

class Rosenbrock():
    """
    see scipy.optimize.rosen
    """
    def __init__(self, dim, k=100):
        self.dim = dim
        self.k = k
    def f(self, x):
        x = np.asarray(x)
        return np.sum(self.k * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, axis=0)
    def grad(self, x):
        x = np.asarray(x)
        g = np.empty(self.dim)
        g[:-1] = 2*(x[:-1]-1) - 4*self.k*(x[1:]-x[:-1]**2)*x[:-1]
        g[-1] = 0
        g[1:] += 2*self.k*(x[1:]-x[:-1]**2)
        return g


def plot():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(111, projection='3d')
    x = np.linspace(-3,3)
    y = np.linspace(-2,6)
    x,y = np.meshgrid(x,y)
    ros = Rosenbrock(dim=2, k=10)
    z = ros.f([x,y])
    ax.plot_surface(x, y, z, cmap=cm.gnuplot_r)
    ax.view_init(22, 120)
    plt.draw()
    plt.show()

if __name__ == "__main__":
    import sys
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    import utility
    ros = Rosenbrock(dim=2, k=10)
    x = np.linspace(-2,2, 10)
    y = np.linspace(-2,2, 10)
    x,y = np.meshgrid(x,y)
    xs = np.stack([x.flat,y.flat],axis=1)
    maxerr = utility.derivativetest.test_grad(ros.f, ros.grad, xs)
    print(f"{maxerr=}")