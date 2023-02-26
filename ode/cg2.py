import numpy as np
from SIMOS_GM.ode import classes

#------------------------------------------------------------------
class Cg2P(classes.Method):
    def __init__(self, method = 'cg2-dg1'):
        super().__init__(error_type = "H1")
        self.name = type(self).__name__ + f"({method})"
        self.method = method
    def run_forward(self, t, app):
        apphasl = hasattr(app,'l')
        if apphasl:
            l1 = np.asarray(app.l(t)).T
            tm = 0.5 * (t[:-1] + t[1:])
            l2 = np.asarray(app.l(tm)).T
        u = app.u0
        # print(f"{u=} {u.shape=}")
        dim, nt = len(u), t.shape[0]
        u1 = np.empty(shape=(nt, dim), dtype=u.dtype)
        u2 = np.empty(shape=(nt - 1, dim), dtype=u.dtype)
        bloc = np.empty(2*dim, dtype=u.dtype)
        Aloc = np.empty((2*dim,2*dim), dtype=u.dtype)
        Mloc = np.eye(dim, dtype=u.dtype)

        u1[0] = u
        for it in range(nt-1):
            # compute u_ap[it+1] and u2[it]
            dt = t[it+1]-t[it]
            assert(dt>0)
            u0 = u1[it]
            f0 = np.asarray(app.f(u0), dtype=u.dtype)
            A0 = np.asarray(app.df(u0), dtype=u.dtype).reshape(dim, dim)
            bloc[:dim] = dt * f0
            bloc[dim:] = 0.5 * dt * f0
            if apphasl:
                bloc[:dim] += dt * 0.5 * (l1[it]+l1[it+1])
                if self.method.split('-')[1]=="2dg0":
                    bloc[dim:] += dt* ((3/8)*l1[it] + (1/8)*l1[it+1])
                else:
                    bloc[dim:] += dt * (l1[it]/3+l1[it+1]/6)
                ll = 4*(l2[it]-0.5*(l1[it]+l1[it+1]))
                bloc[:dim] += dt  * ll/6
                bloc[dim:] += dt * ll/12
            if self.method == 'cg2-dg1':
                alpha = 0
                Aloc[:dim, :dim] = Mloc - (dt/2)*(1+alpha/3)*A0
                Aloc[:dim, dim:] = (dt/6)*A0
                Aloc[dim:, :dim] = Mloc/2*(1+alpha/3) - (dt/6)*(1+alpha/2)*A0
                Aloc[dim:, dim:] = Mloc/6 - (dt/12)*A0
            elif self.method == 'cg2-2dg0':
                Aloc[:dim, :dim] = Mloc - (dt / 2) * A0
                Aloc[:dim, dim:] = (dt / 6) * A0
                Aloc[dim:, :dim] = Mloc / 2 - (dt / 4) * A0
                Aloc[dim:, dim:] = Mloc / 4 - (dt / 12) * A0
            elif self.method == '2cg1-dg1':
                Aloc[:dim, :dim] = Mloc - (dt / 2) * A0
                Aloc[:dim, dim:] = (dt / 8) * A0
                Aloc[dim:, :dim] = Mloc / 2 - (dt / 6) * A0
                Aloc[dim:, dim:] = Mloc / 8 - (dt / 16) * A0
            elif self.method == '2cg1-2dg0':
                Aloc[:dim, :dim] = Mloc - (dt / 2) * A0
                Aloc[:dim, dim:] = (dt / 8) * A0
                Aloc[dim:, :dim] = Mloc / 2 - (dt / 8) * A0
                Aloc[dim:, dim:] = Mloc / 4 - (dt / 16) * A0
            elif self.method == 'cg2+-dg0':
                Aloc[:dim, :dim] = Mloc - (dt / 2) * A0
                Aloc[:dim, dim:] = (dt / 6) * A0
                if it:
                    Aloc[dim:, :dim] = Mloc
                    Aloc[dim:, dim:] = Mloc
                else:
                    Aloc[dim:, :dim] = Mloc/2 - (dt/6)*A0
                    Aloc[dim:, dim:] = Mloc/6 - (dt/12)*A0
            else:
                raise ValueError(f"unknown {self.method=}")
            bloc[:dim] += Aloc[:dim, :dim] @ u0
            if self.method == 'cg2+-dg0' and it:
                fac = (t[it]-t[it-1])/dt
                bloc[dim:] = (1+fac)*u0 - fac*(u1[it-1] + u2[it-1])
            else:
                bloc[dim:] += Aloc[dim:, :dim] @ u0
            # print(f"{b=}")
            # print(f"{A=}")
            usol = np.linalg.solve(Aloc, bloc)
            # print(f"{usol=}")
            u1[it + 1] = usol[:dim]
            u2[it] = usol[dim:]
        return u1, 0.5*(u1[:-1]+u1[1:]) + 0.25*u2

#------------------------------------------------------------------
    def estimator(self, t, sol_ap, app):
        apphasl = hasattr(app,'l')
        u1, u2 = sol_ap
        assert self.method == 'cg2-dg1'
        nt, dim = u1.shape
        estnl = np.empty(shape=(nt-1), dtype=u1.dtype)
        estap = np.empty(shape=(nt-1), dtype=u1.dtype)
        for it in range(nt-1):
            dt = t[it+1] - t[it]
            df0 = np.asarray(app.df(u1[it]), dtype=u1.dtype)
            f0 = np.asarray(app.f(u1[it]), dtype=u1.dtype)
            f1 = np.asarray(app.f(u1[it+1]), dtype=u1.dtype)
            r1 = f1 - f0 - df0@(u1[it+1]-u1[it])
            fm = app.f(u2[it])
            rm = fm - f0 - df0@(u2[it]-u1[it])
            estnl[it] = dt * ((2 / 3) * np.sum(rm ** 2) + (1 / 6) * np.sum(r1 ** 2))
            r = df0@(4*u2[it]-2*(u1[it]+u1[it+1]))
            estap[it] = dt*np.sum(r*r)/6
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
        # unlucky attempt to vectorize
        # dt = t[1:]-t[:-1]
        # f1 = np.asarray(np.vectorize(f, signature='(n,m)->(n,m)', otypes=[list])(u_ap))
        # f2 = np.asarray(f(u2))
        # df1 = np.asarray(df(u_ap))
        # print(f"{u_ap.shape=} {u2.shape=} {f1.shape=} {f2.shape=} {df1.shape=}")
        # r1 = f1[1:] - f1[:-1] - df1[:,-1]@(u_ap[1:]-u_ap[:,-1])
        # print(f"{r1.shape=}")

#------------------------------------------------------------------
    def compute_error(self, t, sol, dsol, sol_ap):
        u1, u2 = sol_ap
        dim, nt = u1.shape[1], t.shape[0]
        dt = (t[1:]-t[:-1])[:,np.newaxis]
        tm = 0.5*(t[1:]+t[:-1])
        solt = np.asarray(sol(t), dtype=u1.dtype).T
        soltm = np.asarray(sol(tm), dtype=u1.dtype).T
        e1 = np.fabs(solt-u1)
        e2 = np.fabs(soltm - u2)
        errfct = {'err_node': e1, 'err_mp': e2}
        err = {}
        # trapez
        err['L2_tr'] = np.sqrt(0.5*np.sum( e1[1:]**2*dt + e1[:-1]**2*dt, axis=0))
        err['max_node'] = np.amax( np.fabs(e1), axis=0)
        err['max_mid'] = np.amax( np.fabs(e2), axis=0)
        # simpson
        err['L2'] = np.sqrt( np.sum( e1[1:]**2*dt/6 + e1[:-1]**2*dt/6 + dt*e2**2*2/3, axis=0))
        dsolt = np.asarray(dsol(t), dtype=u1.dtype).T
        dsoltm = np.asarray(dsol(tm), dtype=u1.dtype).T
        du2 = (u1[1:]-u1[:-1])/dt
        u22 = 4*u2 - 2*(u1[:-1] + u1[1:])
        if self.method.split('-')[0] in ['cg2', 'cg2+']:
            du2l = du2 + u22/dt
            du2r = du2 - u22/dt
        elif self.method.split('-')[0] in ['2cg1']:
            du2l = du2 + 0.5*u22/dt
            du2r = du2 - 0.5*u22/dt
        else:
            raise ValueError(f"{self.method.split('-')[0]=}")
        du2 -= dsoltm
        du2l -= dsolt[:-1]
        du2r -= dsolt[1:]
        errfct['err_der'] = np.fabs(du2)
        err['H1_mp'] = np.sqrt(np.sum( du2**2*dt, axis=0))
        err['H1'] = np.sqrt(np.sum( du2**2*dt*2/3 + du2l**2*dt/6 + du2r**2*dt/6, axis=0))
        return errfct, err


#------------------------------------------------------------------
if __name__ == "__main__":
    pass

