import classes


#------------------------------------------------------------------
class Cubic(classes.Application):
    def __init__(self, u0=1/100, scale=500):
        super().__init__(u0=u0, T=1)
        self.f = lambda u: [scale*u**2*(1-u)]
        self.df = lambda u: [scale*(2*u-6*u**2)]

#------------------------------------------------------------------
class Lorenz(classes.Application):
    def __init__(self, T=30, sigma=10, rho=28, beta=8/3):
        # super().__init__(u0=[-10, -4.45, 35.1], T=20)
        super().__init__(u0=[1,0,0], T=T)
        self.f = lambda u: [sigma*(u[1]-u[0]), rho*u[0]-u[1]-u[0]*u[2],u[0]*u[1]-beta*u[2]]
        self.df = lambda u: [[-sigma,sigma,0], [rho-u[2],-1,-u[0]], [u[1],u[0],-beta]]
    def plot(self, fig, t, u, axkey=(1,1,1), label_u=r'$u_{\delta}$', label_ad=''):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        # print(f"{axkey=}")
        ax = fig.add_subplot(*axkey, projection='3d')
        x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label=label_ad)
        ax.legend()
        # ax.plot_surface(x, y, z, cmap=cm.gnuplot_r, alpha=0.7)
        # ax.contour(x, y, z, np.linspace(0,20,10), offset=-1, linewidths=2, cmap=cm.gray)
        ax.view_init(26, 130)
        plt.draw()

