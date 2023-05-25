import numpy as np  
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print("***", SCRIPT_DIR)
from SIMOS_GM.ode import classes
# import classes

#==================================================================
class Cg1(classes.Method):
#==================================================================
    """
    cgp on [-1,1]
    """
    def __init__(self, alpha=0):
        # alpha=0: CN alpha=1: Euler
        self.alpha = alpha
        super().__init__(error_type = "H1")
#------------------------------------------------------------------
    def run_forward(self, t, app, linearization=None, lintrandom=None, q=0):
        assert linearization is  None
        u_ic = app.u_zero()
        dim, nt = app.dim, t.shape[0]
        apphasl = hasattr(app,'l')
        dt = t[1:] - t[:-1]
        if apphasl:
            tm = 0.5*(t[1:]+t[:-1])
            lint = np.asarray(app.l(tm)).T
            # print(f"{(tm+0.5*self.int_x[:,np.newaxis]*dt).shape=} {lint.shape=}")
        u_node = np.empty(shape=(nt, dim), dtype=app.dtype)
        bloc = np.empty(shape=(dim), dtype=u_node.dtype)
        Aloc = np.zeros((dim, dim), dtype=u_node.dtype)
        if hasattr(app, 'M'):
            M = app.M
        else:
            M = np.eye(dim, dtype=app.dtype)
        u_node[0] = u_ic
        for it in range(nt-1):
            dt = t[it+1]-t[it]
            assert(dt>0)
            bloc.fill(0)
            utilde = u_node[it]
            f0 = np.asarray(app.f(utilde), dtype=u_node.dtype)
            bloc[0] = dt * f0
            A0 = np.asarray(app.df(utilde), dtype=u_node.dtype).reshape(dim, dim)
            # A0 = app.df(utilde)
            Aloc = 2*M - dt*A0
            if self.alpha:
                Aloc -= self.alpha*dt*A0
            if apphasl:
                bloc += 0.5*dt*lint[it]
            if lintrandom is not None:
                bloc += lintrandom[it]*utilde
                Aloc += q**2*M*dt
            usol = np.linalg.solve(Aloc, bloc)
            u_node[it + 1] = 2*usol + u_node[it]
        return u_node
# ------------------------------------------------------------------
    def interpolate(self, t, u_node, mean=False):
        return u_node
