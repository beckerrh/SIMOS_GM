import numpy as np

#==================================================================
class Method():
#==================================================================
    def __init__(self, error_type="H1"):
        self.error_type = error_type
        self.name = type(self).__name__
#------------------------------------------------------------------
    def interpolate(self, t, sol_ap):
        return sol_ap

#==================================================================
class Application():
# ==================================================================
    def __init__(self, u0=None, T=None):
        if u0:
            if isinstance(u0, list) or isinstance(u0, np.ndarray):
                self.u0 = u0
            else:
                self.u0 = [u0]
            self.u0 = np.asarray(self.u0, dtype=np.float64)
        if T: self.T = T
        self.name = type(self).__name__
    def plot(self, fig, t, u, axkey=(1,1,1), label_u=r'$u_{\delta}$', label_ad=''):
        ax = fig.add_subplot(*axkey)
        dim = u.shape[1]
        for j in range(dim):
            ax.plot(t, u[:, j], '-', label=f'{label_u}({j:1d}) {label_ad}')
        ax.legend()
        ax.grid()

#==================================================================
class Functional():
# ==================================================================
    def __init__(self):
        self.name = type(self).__name__

#==================================================================
class FunctionalEndTime(Functional):
# ==================================================================
    def __init__(self, k):
        super().__init__()
        self.k = k
    def lT(self, u):
        return u[self.k]
    def lT_prime(self, u):
        v = np.zeros_like(u)
        v[self.k] = 1
        return v

#==================================================================
class FunctionalMean(Functional):
# ==================================================================
    def __init__(self, k):
        super().__init__()
        self.k = k
    def l(self, t, u):
        return u[self.k]
    def l_prime(self, t, u):
        v = np.zeros_like(u)
        v[self.k] = 1
        return v
