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
    def __init__(self, u0=None, T=None, dim=None):
        self.dtype = np.float64
        if dim is not None: self.dim = dim
        if u0:
            if isinstance(u0, list) or isinstance(u0, np.ndarray):
                self.u0 = u0
            else:
                self.u0 = [u0]
            self.u0 = np.asarray(self.u0, dtype=self.dtype)
            if dim is None: self.dim = len(self.u0)
            else: assert self.dim==len(self.u0)
        if T: self.T = T
        self.name = type(self).__name__
    def u_zero(self): return self.u0
    def plot(self, fig, t, u, axkey=(1,1,1), label_u=r'$u_{\delta}$', label_ad='', title=''):
        ax = fig.add_subplot(*axkey)
        ax.set_title(title)
        dim = u.shape[1]
        for j in range(dim):
            ax.plot(t, u[:, j], '-', label=f'{label_u}({j:1d}) {label_ad}')
        ax.legend()
        ax.grid()
    def setParameter(self, p): self.p = p

#==================================================================
class Functional():
# ==================================================================
    def __init__(self):
        self.name = type(self).__name__

#==================================================================
class FunctionalEndTime(Functional):
# ==================================================================
    def __init__(self, k=0):
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
    def __init__(self, k=0, t0=-np.inf, t1=np.inf):
        super().__init__()
        self.k, self.t0, self.t1 = k, t0, t1
    def l(self, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        return u[self.k]
    def l_prime(self, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        v = np.zeros_like(u)
        v[self.k] = 1
        return v


#==================================================================
class FunctionalTimePoint(Functional):
# ==================================================================
    def __init__(self, method, t0, k=0):
        super().__init__()
        self.k, self.t0, self.method = k, t0, method
    def ldelta(self, t, u_app):
        i = np.searchsorted(t, self.t0)
        print(f"{self.t0} {t[i]=}")
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        return self.method.evaluate(u_app, self.k)
    def ldelta_prime(self, b, t, u):
        # if t<self.t0 or t>self.t1: return np.zeros_like(u)
        v = np.zeros_like(u)
        v[self.k] = 1
        return v
