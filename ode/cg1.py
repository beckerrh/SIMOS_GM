import numpy as np
from SIMOS_GM.ode import classes

#==================================================================
class Cg1P(classes.Method):
#==================================================================
    def __init__(self, alpha=0):
        super().__init__(error_type = "H1")
        self.k = 1
        self.alpha = alpha
#------------------------------------------------------------------
    def run_forward(self, t, app):
        apphasl = hasattr(app,'l')
        if apphasl:
            l1 = np.asarray(app.l(t)).T
            tm = 0.5 * (t[:-1] + t[1:])
            dl = np.asarray(app.l(tm)).T
            dl = 4*(dl-0.5*(l1[:-1]+l1[1:]))
        u_ic = app.u0
        dim, nt = len(u_ic), t.shape[0]
        u_ap = np.empty(shape=(nt, dim), dtype=u_ic.dtype)
        bloc = np.empty(dim, dtype=u_ic.dtype)
        Aloc = np.empty((dim,dim), dtype=u_ic.dtype)
        M = np.eye(dim, dtype=u_ic.dtype)
        u_ap[0] = u_ic
        for it in range(nt-1):
            # compute u_ap[it+1] and u2[it]
            dt = t[it+1]-t[it]
            assert(dt>0)
            utilde = u_ap[it]
            f0 = np.asarray(app.f(utilde), dtype=u_ic.dtype)
            A0 = np.asarray(app.df(utilde), dtype=u_ic.dtype).reshape(dim, dim)
            bloc[:dim] = dt * f0
            if apphasl:
                bloc[:dim] += dt * 0.5 * (l1[it]+l1[it+1])
                bloc[:dim] += dt * dl[it] / 6
            alpha = self.alpha*np.fmin(dt,1)
            Aloc[:dim, :dim] = M - (0.5+alpha)*dt*A0
            bloc[:dim] += Aloc[:dim, :dim] @ utilde
            usol = np.linalg.solve(Aloc, bloc)
            u_ap[it + 1] = usol[:dim]
        return u_ap
#------------------------------------------------------------------
    def compute_functional(self, t, u_ap, func):
        if not isinstance(u_ap, np.ndarray):
           u_ap = np.array(u_ap(t)).T
        funchasl = hasattr(func, 'l')
        funchaslT = hasattr(func, 'lT')
        assert int(funchasl) + int(funchaslT) == 1
        if funchaslT:
            return func.lT(u_ap[-1])
        fl_vec = np.vectorize(func.l, signature="(n),(k,n)->(n)", otypes=[float])
        l1 = fl_vec(t, u_ap.T)
        # print(f"{t.shape=} {u_ap.shape=} {l1.shape=}")
        tm = 0.5 * (t[:-1] + t[1:])
        um = 0.5 * (u_ap[:-1] + u_ap[1:])
        lm = fl_vec(tm, um.T)
        dt = t[1:]-t[:-1]
        return  np.sum(dt*((1/6)*l1[:-1]+(1/6)*l1[1:] + (2/3)*lm))
#------------------------------------------------------------------
    def run_backward(self, t, u_ap, app, func):
        nt, dim = u_ap.shape
        funchasl = hasattr(func,'l_prime')
        funchaslT = hasattr(func,'lT_prime')
        if funchasl:
            lprime_vec = np.vectorize(func.l_prime, signature="(n),(k,n)->(k,n)", otypes=[float])
            l1 = lprime_vec(t, u_ap.T).T
            # l1 = np.asarray(func.l_prime(t, u_ap))
            tm = 0.5 * (t[:-1] + t[1:])
            um = 0.5 * (u_ap[:-1] + u_ap[1:])
            # dl = np.asarray(func.l_prime(tm, um))
            dl = lprime_vec(tm, um.T).T
            # print(f"{l1.shape=} {dl.shape=}")
            dl = 4*(dl-0.5*(l1[:-1]+l1[1:]))
        z_ap = np.empty(shape=(nt-1, dim), dtype=u_ap.dtype)
        bloc = np.empty(dim, dtype=u_ap.dtype)
        Aloc = np.empty((dim,dim), dtype=u_ap.dtype)
        M = np.eye(dim, dtype=u_ap.dtype)
        b0 = np.zeros_like(bloc)
        b1 = np.zeros_like(bloc)
        if funchaslT:
            zT = np.asarray(func.lT_prime(u_ap[nt - 1]), dtype=u_ap.dtype)
        else:
            zT = np.zeros_like(bloc)
        for it in range(nt-2,-1,-1):
            dt = t[it+1]-t[it]
            # utilde = u_ap[it]
            # umean = 0.5*(u_ap[it]+u_ap[it+1])
            # Aap = app.df(u_ap[it]) + app.df(0.75*u_ap[it]+0.25*u_ap[it+1]) - app.df(1.25*u_ap[it]-0.25*u_ap[it+1])
            A0 = np.asarray(app.df(u_ap[it]), dtype=u_ap.dtype).reshape(dim, dim).T
            A0 += np.asarray(app.df(0.75*u_ap[it]+0.25*u_ap[it+1]), dtype=u_ap.dtype).reshape(dim, dim).T
            A0 -= np.asarray(app.df(1.25*u_ap[it]-0.25*u_ap[it+1]), dtype=u_ap.dtype).reshape(dim, dim).T
            if funchasl:
                b1[:] = dt * (l1[it]/6 + l1[it+1]/3 + dl[it]/12)
            if it==nt-2:
                bloc[:] = zT[:]
            else:
                bloc[:] = b0[:]
            # print(f"{it=} {b1=}")
            bloc[:] += b1[:]
            alpha = self.alpha*np.fmin(dt,1)
            Aloc[:dim, :dim] = M - (0.5+alpha)*dt*A0
            zsol = np.linalg.solve(Aloc, bloc)
            z_ap[it] = zsol[:dim]
            b0[:] = (M + (0.5 - alpha) * dt * A0) @ z_ap[it]
            if funchasl:
                b0[:] += dt * (l1[it]/3 + l1[it+1]/6 + dl[it]/12)
        return b0, z_ap, zT
#------------------------------------------------------------------
    def compute_functional_dual(self, t, z_sol, app):
        z0, z_ap, zT = z_sol
        val = np.dot(app.u0, z0)
        if hasattr(app,'l'):
            l1 = app.l(t)
            tm = 0.5 * (t[:-1] + t[1:])
            lm = app.l(tm)
            dt = t[1:] - t[:-1]
            val += np.sum(dt*((1/6)*l1[:-1]+(1/6)*l1[1:] + (2/3)*lm)*z_ap)
        return val
#------------------------------------------------------------------
    def estimator(self, t, u1, app):
        apphasl = hasattr(app,'l')
        nt, dim = u1.shape
        estnl = np.empty(shape=(nt-1), dtype=u1.dtype)
        estap = np.empty(shape=(nt-1), dtype=u1.dtype)
        if apphasl:
            lt = np.asarray(app.l(t), dtype=u1.dtype).T
            tm = 0.5*(t[1:]+t[:-1])
            lm = np.asarray(app.l(tm), dtype=u1.dtype).T
        for it in range(nt-1):
            dt = t[it+1] - t[it]
            df0 = np.asarray(app.df(u1[it]), dtype=u1.dtype)
            f0 = np.asarray(app.f(u1[it]), dtype=u1.dtype)
            f1 = np.asarray(app.f(u1[it+1]), dtype=u1.dtype)
            fm = np.asarray(app.f(0.5*(u1[it]+u1[it+1])), dtype=u1.dtype)
            r1 = f1 - f0 - df0@(u1[it+1]-u1[it])
            rm = fm - f0 - 0.5*df0@(u1[it+1]-u1[it])
            estnl[it] = dt * (1 / 6) * np.sum(r1 ** 2)
            estnl[it] += dt * (2 / 3) * np.sum(rm ** 2)
            du = u1[it+1]-u1[it]
            r = df0@du
            # print(f"{it=} {u1[it]=} {np.sum(du*du)=} {np.sum(r*r)=}")
            alpha = self.alpha*np.fmin(dt,1)
            r2 = 3*alpha*r.copy()
            if apphasl:
                r += lt[it+1]-lt[it]
                r2 += 2*lm[it] - lt[it+1]-lt[it]
            estap[it] = dt*np.sum(r*r)/12 + dt*np.sum(r2*r2)/9


        # estnlnew = np.empty(shape=(nt-1), dtype=u1.dtype)
        # estapnew = np.empty(shape=(nt-1), dtype=u1.dtype)
        # dt = t[1:] - t[:-1]
        # # f_vec = np.vectorize(app.f, signature="(k,n)->(k,n)")
        # f_vec = np.vectorize(app.f, signature="(k,n)->(k,n)", otypes=[float])
        # df_vec = np.vectorize(app.df, signature="(k,n)->(k,k,n)", otypes=[float])
        # # print(f"{u1.shape=} {app.f} {app.f.__name__}")
        # f1 = f_vec(u1).T
        # # print(f"{f1=}")
        # # print(f"{f1.shape=}")
        # um = 0.5*(u1[1:]+u1[:-1])
        # fm = f_vec(um.T).T
        # df0 = df_vec(u1.T).T
        # du = u1[1:] - u1[:-1]
        # r1 = f1[1:] - f1[:-1] - df0 @ (du)
        # rm = fm - f1[:-1] - 0.5 * df0 @ (u1[1:] - u1[:-1])
        # estnlnew =  (1 / 6) * np.sum(dt *r1 ** 2)
        # estnlnew += (2 / 3) * np.sum(dt * rm ** 2)
        # r = df0 @ du
        # # alpha = dt
        # estapnew =  np.sum(dt *r * r) / 6 +  np.sum(dt ** 2 *du * du) / 3
        # if apphasl:
        #     dl = lt[1:] - lt[:-1]
        #     estapnew +=  0.25 * np.sum(dt *dl * dl)

        # assert np.allclose(estap, estapnew)
        # assert np.allclose(estnl, estnlnew)
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
# ------------------------------------------------------------------
    def estimator_dual(self, t, u_ap, z_sol, app, func):
        z0, z_ap, zT = z_sol
        funchasl = hasattr(func,'l_prime')
        if funchasl:
            lprime_vec = np.vectorize(func.l_prime, signature="(n),(k,n)->(k,n)", otypes=[float])
            l1 = lprime_vec(t, u_ap.T).T
            # l1 = np.asarray(func.l_prime(t, u_ap))
            tm = 0.5 * (t[:-1] + t[1:])
            um = 0.5 * (u_ap[:-1] + u_ap[1:])
            lm = lprime_vec(tm, um.T).T
            # dl = np.asarray(func.l_prime(tm, um))
            dl = lprime_vec(tm, um.T).T
            # print(f"{l1.shape=} {dl.shape=}")
            dl = 4*(dl-0.5*(l1[:-1]+l1[1:]))
        nt, dim = u_ap.shape
        estnl = np.empty(shape=(nt-1), dtype=u_ap.dtype)
        estap = np.empty(shape=(nt-1), dtype=u_ap.dtype)
        for it in range(nt - 1):
            dt = t[it + 1] - t[it]
            Am = np.asarray(app.df(0.5 * u_ap[it] + 0.5 * u_ap[it + 1]), dtype=u_ap.dtype).reshape(dim, dim).T
            A0 = np.asarray(app.df(u_ap[it]), dtype=u_ap.dtype).reshape(dim, dim).T
            A0 += np.asarray(app.df(0.75 * u_ap[it] + 0.25 * u_ap[it + 1]), dtype=u_ap.dtype).reshape(dim, dim).T
            A0 -= np.asarray(app.df(1.25 * u_ap[it] - 0.25 * u_ap[it + 1]), dtype=u_ap.dtype).reshape(dim, dim).T
            estnl[it] = dt*np.sum((A0-Am)@z_ap[it])
            # estap[it] = dt**3*np.sum((A0@z_ap[it])**2)/36
            if it<nt-2:
                r = z_ap[it]-z_ap[it+1]
            else:
                r = z_ap[it] - zT
            # no idea for the factor 1/36
            estap[it] = np.sum(r ** 2) * dt * 2/ 9/36
            if funchasl:
                # estap[it] +=  dt*2*np.sum((l1[it+1]-l1[it])**2)/36
                r = lm[it] - 0.5 * (l1[it] - l1[it + 1])
                # no idea for the factor 1/64
                estap[it] += np.sum(r ** 2) * dt ** 3 * 2 / 81 / 64

        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}

#------------------------------------------------------------------
#------------------------------------------------------------------
    def compute_error(self, t, sol_ex, dsol_ex, u1):
        dim, nt = u1.shape[1], t.shape[0]
        dt = (t[1:]-t[:-1])[:,np.newaxis]
        tm = 0.5*(t[1:]+t[:-1])
        solt = np.asarray(sol_ex(t), dtype=u1.dtype).T
        soltm = np.asarray(sol_ex(tm), dtype=u1.dtype).T
        e1 = np.fabs(solt-u1)
        e2 = np.fabs(soltm - 0.5*(u1[1:]+u1[:-1]))
        errfct = {'err_node': e1, 'err_mp': e2}
        err = {}
        # trapez
        err['L2_nod'] = np.sqrt(0.5*np.sum( e1[1:]**2*dt + e1[:-1]**2*dt, axis=0))
        err['max_nod'] = np.amax( np.fabs(e1), axis=0)
        err['max_mid'] = np.amax( np.fabs(e2), axis=0)
        # simpson
        err['L2'] = np.sqrt( np.sum( e1[1:]**2*dt/6 + e1[:-1]**2*dt/6 + dt*e2**2*2/3, axis=0))
        dsol = np.asarray(dsol_ex(t), dtype=u1.dtype).T
        dsoltm = np.asarray(dsol_ex(tm), dtype=u1.dtype).T
        # du in midpoint alpha_modif vanishes here
        du = (u1[1:]-u1[:-1])/dt
        err_du = du - dsoltm
        err_dul = du - dsol[:-1]
        err_dur = du - dsol[1:]
        errfct['err_der'] = np.fabs(err_du)
        err['H1_mp'] = np.sqrt(np.sum( err_du**2*dt, axis=0))
        err['H1'] = np.sqrt(np.sum( err_du**2*dt*2/3 + err_dul**2*dt/6 + err_dur**2*dt/6, axis=0))
        # print(f"{err=}")
        return errfct, err
    def interpolate(self, t, u_ap):
        dt = t[1:,np.newaxis]-t[:-1,np.newaxis]
        alpha = self.alpha * np.minimum(dt, 1)
        return u_ap, 0.5*(u_ap[:-1]+u_ap[1:]) + alpha*dt*(3/2)*(u_ap[1:]-u_ap[:-1])
#------------------------------------------------------------------
    def interpolate_dual(self, t, z_ap):
        z0, z_m, zT = z_ap
        z_n = np.empty(shape=(len(t), z_m.shape[1]))
        z_n[0] = z0
        z_n[1:-1] = 0.5*(z_m[:-1] + z_m[1:])
        z_n[-1] = zT
        return z_n, z_m
#==================================================================
class Cg1D(classes.Method):
    """
    Approx-space: constant
    Test-space: P1 modified
    """
#==================================================================
    def __init__(self, alpha=0.1):
        super().__init__(error_type = "L2")
        self.alpha = alpha
#------------------------------------------------------------------
    def run_forward(self, t, app):
        apphasl = hasattr(app,'l')
        if apphasl:
            l1 = np.asarray(app.l(t)).T
            tm = 0.5 * (t[:-1] + t[1:])
            l2 = np.asarray(app.l(tm)).T
            l2 = 4 * (l2 - 0.5 * (l1[:-1] + l1[1:]))
        u_ic = app.u0
        dim, nt = len(u_ic), t.shape[0]
        u_ap = np.empty(shape=(nt-1, dim), dtype=u_ic.dtype)
        bloc = np.empty(dim, dtype=u_ic.dtype)
        Aloc = np.empty((dim,dim), dtype=u_ic.dtype)
        M = np.eye(dim, dtype=u_ic.dtype)
        for it in range(nt-1):
            dt = t[it+1]-t[it]
            assert(dt>0)
            if it==0:
                utilde = u_ic.copy()
                bold = u_ic.copy()
            else:
                utilde = u_ap[it-1]
            f0 = np.asarray(app.f(utilde), dtype=u_ic.dtype)
            A0 = np.asarray(app.df(utilde), dtype=u_ic.dtype).reshape(dim, dim)
            alpha = self.alpha*np.fmin(dt,1)
            b0 = 0.5 * dt * (f0 - A0@utilde)
            b0 = (0.5+alpha) * dt * (f0 - A0@utilde)
            bimp = bold + b0
            if apphasl:
                bimp += (1/3)*dt*l1[it] + (1/6)*dt*l1[it+1] +(1/12)*l2[it]
                # bimp += 0.5*alpha*(l1[it]+l1[it+1]) + alpha/5*l2[it]
            Aloc[:dim, :dim] = M - (0.5+alpha)*dt*A0
            bloc[:dim] = bimp
            usol = np.linalg.solve(Aloc, bloc)
            u_ap[it] = usol[:dim]
            bold = b0
            bold = (0.5-alpha) * dt * (f0 - A0@utilde)
            if apphasl:
                bold += (1 / 6) * dt * l1[it] + (1 / 3) * dt * l1[it + 1] +(1/12)*l2[it]
                # bold -= 0.5*alpha*(l1[it]+l1[it+1])+ alpha/5*l2[it]
            bold += (M + (0.5-alpha)*dt*A0)@u_ap[it]
        return u_ic, u_ap, bold
#------------------------------------------------------------------
    def estimator(self, t, sol_ap, app):
        apphasl = hasattr(app,'l')
        u_ic, u_ap, u_T = sol_ap
        nt, dim = t.shape[0], u_ap.shape[1]
        estnl = np.empty(shape=(nt-1), dtype=u_ap.dtype)
        estap = np.empty(shape=(nt-1), dtype=u_ap.dtype)
        if apphasl:
            l1 = np.asarray(app.l(t), dtype=u_ap.dtype).T
            tm = 0.5 * (t[:-1] + t[1:])
            lm = np.asarray(app.l(tm), dtype=u_ap.dtype).T
        for it in range(nt-1):
            dt = t[it+1] - t[it]
            if it==0:
                utilde = u_ic.copy()
            else:
                utilde = u_ap[it-1]
            f0 = np.asarray(app.f(utilde), dtype=u_ic.dtype)
            A0 = np.asarray(app.df(utilde), dtype=u_ic.dtype).reshape(dim, dim)
            f1 = np.asarray(app.f(u_ap[it]), dtype=u_ap.dtype)
            r1 = f1 - f0 - A0@(u_ap[it] - utilde)
            estnl[it] = dt * np.sum(r1 ** 2)
            alpha = self.alpha*np.fmin(dt,1)
            # point-wise interpolator
            # r = f0 + A0@(u_ap[it] - utilde)
            # if apphasl:
            #     r += lm[it]
            # estap[it] = np.sum(r ** 2) * dt **3 * 4 / 9
            if it == nt-2: r = u_ap[it]-u_T
            else: r = u_ap[it]-u_ap[it+1]
            # no idea for the factor 1/36
            estap[it] = np.sum(r ** 2) * dt * 2/ 9/36 * (1+4*alpha/(2*alpha+1))
            if apphasl:
                r = lm[it]-0.5*(l1[it]-l1[it+1])
                # no idea for the factor 1/64
                estap[it] += np.sum(r ** 2) * dt **3 * 2 / 81 /64
            # estap[it] = np.sum(f1 ** 2)*dt**3/36
            # if apphasl:
            #     dl = lt[it + 1] - lt[it]
            #     estap[it] += dt * (1/36) * np.sum(dl * dl)
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
#------------------------------------------------------------------
    def compute_error(self, t, sol_ex, dsol_ex, sol_ap):
        u_ic, u_ap, u_T = sol_ap
        dim, nt = u_ap.shape[1], t.shape[0]
        dt = (t[1:]-t[:-1])[:,np.newaxis]
        tm = 0.5*(t[1:]+t[:-1])
        solt = np.asarray(sol_ex(t), dtype=u_ap.dtype).T
        soltm = np.asarray(sol_ex(tm), dtype=u_ap.dtype).T
        em = np.fabs(soltm-u_ap)
        en = np.empty(shape=(nt,dim), dtype=u_ap.dtype)
        en[0] = np.fabs(solt[0]-u_ap[0])
        en[1:-1] = np.fabs(solt[1:-1] - 0.5*(u_ap[1:]+u_ap[:-1]))
        en[-1] = np.fabs(solt[-1]-u_ap[-1])
        errfct = {'err_node': en, 'err_mp': em}
        err = {}
        # trapez
        err['L2_node'] = np.sqrt(0.5*np.sum( en[1:]**2*dt + en[:-1]**2*dt, axis=0))
        err['L2_mid'] = np.sqrt(np.sum( dt*em**2, axis=0))
        err['max_node'] = np.amax( np.fabs(en), axis=0)
        err['max_mid'] = np.amax( np.fabs(em), axis=0)
        # simpson
        err['L2'] = np.sqrt( np.sum( en[1:]**2*dt/6 + en[:-1]**2*dt/6 + dt*em**2*2/3, axis=0))
        # dsol = np.asarray(dsol_ex(t), dtype=u_ap.dtype).T
        # dsoltm = np.asarray(dsol_ex(tm), dtype=u_ap.dtype).T
        # du = (u_ap[1:]-u_ap[:-1])/dt
        # err_du = du - dsoltm
        # err_dul = du - dsol[:-1]
        # err_dur = du - dsol[1:]
        # errfct['err_der'] = np.fabs(err_du)
        # err['H1_mp'] = np.sqrt(np.sum( err_du**2*dt, axis=0))
        # err['H1'] = np.sqrt(np.sum( err_du**2*dt*2/3 + err_dul**2*dt/6 + err_dur**2*dt/6, axis=0))
        return errfct, err
#------------------------------------------------------------------
    def interpolate(self, t, sol_ap):
        u_ic, u_ap, u_T = sol_ap
        dim, nt = u_ap.shape[1], t.shape[0]
        u_node = np.empty(shape=(nt,dim), dtype=u_ap.dtype)
        u_node[-1] = u_T
        u_node[0] = u_ap[0]
        u_node[1:-1] = 0.5*(u_ap[1:] + u_ap[:-1])
        return u_node, u_ap
#------------------------------------------------------------------
if __name__ == "__main__":
    import cgp
    method = Cg1P()
    # test_alpha_wave()
    cgp.test_dual_wave(method=method)
