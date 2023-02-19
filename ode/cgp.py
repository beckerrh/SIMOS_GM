import numpy as np
import classes

#==================================================================
class CgP(classes.Method):
#==================================================================
    """
    Attempt to do cg1 on [-1,1]
    """
    def __init__(self, k=1):
        super().__init__(error_type = "H1")
        assert k>=1
        self.k = k
        self.int_x, self.int_w = np.polynomial.legendre.leggauss(k+2)
        self.int_n = len(self.int_w)
        self.psi, self.phi = [], []
        for i in range(k):
            p = np.polynomial.legendre.Legendre.basis(deg=i, domain=[-1, 1])
            self.psi.append(p)
            q = p.integ()
            self.phi.append(q-q(-1))
            assert self.phi[-1].deriv() == self.psi[-1]
        import matplotlib.pyplot as plt
        t = np.linspace(-1,1)
        fig = plt.figure(figsize=plt.figaspect(2))
        ax = plt.subplot(211)
        for i in range(len(self.phi)):
            plt.plot(t, self.psi[i](t), '-', label=r"$\psi_{:2d}$".format(i))
        plt.legend()
        plt.grid()
        ax = plt.subplot(212)
        for i in range(len(self.phi)):
            plt.plot(t, self.phi[i](t), '-x', label=r"$\phi_{:2d}$".format(i))
        plt.legend()
        plt.grid()
        plt.show()
        self.M = np.zeros(shape=(k,k))
        self.T = np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                for ii in range(self.int_n):
                    self.M[i,j] += self.int_w[ii]* self.psi[i](self.int_x[ii])* self.phi[j](self.int_x[ii])
                    self.T[i, j] += self.int_w[ii] * self.psi[i](self.int_x[ii]) * self.psi[j](self.int_x[ii])
        self.diagT = np.diagonal(self.T)[1:]
        self.coefM = np.diagonal(self.M, offset=-1)
        self.int_psi_weights = np.empty(shape=(len(self.psi),self.int_n))
        self.int_psi = np.empty(shape=(len(self.psi),self.int_n))
        self.int_phi = np.empty(shape=(len(self.psi),self.int_n))
        for i in range(k):
            for ii in range(self.int_n):
                self.int_psi_weights[i,ii] = self.psi[i](self.int_x[ii])*self.int_w[ii]
                self.int_psi[i, ii] = self.psi[i](self.int_x[ii])
                self.int_phi[i, ii] = self.phi[i](self.int_x[ii])
        self.phi_mid = np.array([self.phi[ik](0) for ik in range(self.k)])
        self.psi_mid = np.array([self.psi[ik](0) for ik in range(self.k)])
        # print(f"{self.diagT=}")
        # print(f"{self.coefM=}")
        # print("\nM")
        # for i in range(k):
        #     for j in range(k):
        #         print(f"{self.M[i,j]:10.6f}",end='')
        #     print()
        # print("\nA",end='')
        # for i in range(k):
        #     for j in range(k):
        #         print(f"{self.A[i, j]:10.6f}",end='')
        #     print()


    #------------------------------------------------------------------
    def run_forward(self, t, app):
        u_ic = app.u0
        dim, nt = len(u_ic), t.shape[0]
        apphasl = hasattr(app,'l')
        dt = t[1:] - t[:-1]
        if apphasl:
            tm = 0.5*(t[1:]+t[:-1])
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
        u_ap = np.empty(shape=(nt, dim), dtype=u_ic.dtype)
        u_coef = np.empty(shape=(nt-1, self.k, dim), dtype=u_ic.dtype)
        # bloc = np.empty(self.k*dim, dtype=u_ic.dtype)
        bloc2 = np.empty(shape=(self.k,dim), dtype=u_ic.dtype)
        Aloc2 = np.zeros((self.k,self.k, dim, dim), dtype=u_ic.dtype)
        # Aloc = np.zeros((self.k*dim,self.k*dim), dtype=u_ic.dtype)
        M = np.eye(dim, dtype=u_ic.dtype)
        u_ap[0] = u_ic
        for it in range(nt-1):
            dt = t[it+1]-t[it]
            assert(dt>0)
            utilde = u_ap[it]
            f0 = np.asarray(app.f(utilde), dtype=u_ic.dtype)
            A0 = np.asarray(app.df(utilde), dtype=u_ic.dtype).reshape(dim, dim)
            # bloc[:] = 0
            # bloc[:dim] = dt * f0
            bloc2.fill(0)
            bloc2[0] = dt*f0
            if apphasl:
                # print(f"{lint[it].shape=} {self.int_psi_weights.shape=} {bloc2.shape=}")
                bloc2 += 0.5*dt*np.einsum('jk,lj->lk', lint[it], self.int_psi_weights)
                # for ik in range(self.k):
                #     bloc[ik*dim:(ik+1)*dim] += 0.5*dt * np.einsum('ij,i', lint[it], self.int_psi_weights[ik])
            # for ik in range(self.k):
            #     for jk in range(self.k):
            #         Aloc[ik*dim: (ik+1)*dim, jk*dim: (jk+1)*dim] = self.T[ik,jk]*M - 0.5*dt *self.M[ik,jk]*A0
            # Aloc[:dim, :dim] = 2*M - dt*A0
            Aloc2[0, 0] = 2*M - dt*A0
            for ik in range(1,self.k):
                # Aloc[ik * dim:(ik + 1) * dim, ik * dim:(ik + 1) * dim] = self.diagT[ik - 1]*M
                # Aloc[(ik-1) * dim : ik * dim, ik * dim : (ik + 1) * dim] = 0.5 * dt * self.coefM[ik - 1]* A0
                # Aloc[ik * dim:(ik + 1) * dim, (ik-1) * dim : ik * dim] = -0.5 * dt * self.coefM[ik - 1] * A0
                Aloc2[ik, ik] = self.diagT[ik - 1] * M
                Aloc2[ik - 1, ik] = 0.5 * dt * self.coefM[ik - 1] * A0
                Aloc2[ik, ik - 1] = -0.5 * dt * self.coefM[ik - 1] * A0
            # assert np.allclose(bloc, bloc2.flat)
            # print(f"{Aloc=}")
            # print(f"{Aloc2.swapaxes(1,2)=}")
            # assert np.allclose(Aloc, Aloc2.swapaxes(1,2).reshape(self.k*dim,self.k*dim))
            # usol = np.linalg.solve(Aloc, bloc2.flat)
            usol2 = np.linalg.solve(Aloc2.swapaxes(1,2).reshape(self.k*dim,self.k*dim), bloc2.flat).reshape((self.k,dim))
            # usol2 = usol.reshape((self.k,dim))
            # usol = np.linalg.solve(Aloc, bloc)
            # for ik in range(self.k):
            #     # print(f"{ik=} {self.phi[ik](1)=}")
            #     assert np.allclose(usol2[ik], usol[ik*dim:(ik+1)*dim])
            #     u_coef[it,ik] = usol[ik*dim:(ik+1)*dim]
            u_coef[it] = usol2
            u_ap[it + 1] = 2*usol2[0] + utilde
            # u_ap[it + 1] = 2 * usol[:dim] + utilde
        return u_ap, u_coef
# ------------------------------------------------------------------
    def interpolate(self, t, u_ap):
        u_node, u_coef = u_ap
        return u_node, u_node[:-1]+np.einsum('ijk,j', u_coef, self.phi_mid)
#------------------------------------------------------------------
    def compute_error(self, t, sol_ex, dsol_ex, u_ap):
        u_node, u_coef = u_ap
        dt = (t[1:]-t[:-1])
        tm = 0.5*(t[1:]+t[:-1])
        solt = np.asarray(sol_ex(t), dtype=u_node.dtype).T
        dsoltm = np.asarray(dsol_ex(tm), dtype=u_node.dtype).T
        udder = np.einsum('ijk,ij->ik', u_coef, self.psi_mid*2/dt[:,np.newaxis])
        e1 = np.fabs(solt-u_node)
        errfct = {'err_node': e1}
        errfct['err_der'] = np.fabs(dsoltm-udder)
        err = {}
        uder2 = np.asarray(dsol_ex(tm + 0.5 * dt * self.int_x[:,np.newaxis]), dtype=u_node.dtype).T
        uint2 = np.asarray(sol_ex(tm + 0.5 * dt * self.int_x[:,np.newaxis]), dtype=u_node.dtype).T
        uder_ap2 = 2*np.einsum('ijk,jl,i->ilk', u_coef, self.int_psi, 1/dt)
        uint_ap2 = u_node[:-1,np.newaxis] + np.einsum('ijk,jl->ilk', u_coef, self.int_phi)
        err['H1'] = 0.5*np.einsum('ilk,l,ik', (uder2-uder_ap2)**2, self.int_w, dt[:,np.newaxis])
        err['L2'] = 0.5*np.einsum('ilk,l,ik', (uint2-uint_ap2)**2, self.int_w, dt[:,np.newaxis])
        err['H1'] = np.sqrt(err['H1'])
        err['L2'] = np.sqrt(err['L2'])
        err['L2_nod'] = np.sqrt(0.5*np.sum( e1[1:]**2*dt[:,np.newaxis] + e1[:-1]**2*dt[:,np.newaxis], axis=0))
        err['max_nod'] = np.amax( np.fabs(e1), axis=0)
        em = np.asarray(sol_ex(tm), dtype=u_node.dtype).T - u_node[:-1]-np.einsum('ijk,j', u_coef, self.phi_mid)
        err['max_mid'] = np.amax( np.fabs(em), axis=0)
        return errfct, err
#------------------------------------------------------------------
    def estimator(self, t, u_ap, app):
        u_node, u_coef = u_ap
        apphasl = hasattr(app,'l')
        nt, dim = u_node.shape
        estnl = np.empty(shape=(nt-1), dtype=u_node.dtype)
        estap = np.empty(shape=(nt-1), dtype=u_node.dtype)
        if apphasl:
            tm = 0.5*(t[1:]+t[:-1])
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
        if apphasl:
            lt = np.asarray(app.l(t), dtype=u_node.dtype).T
            tm = 0.5*(t[1:]+t[:-1])
            lm = np.asarray(app.l(tm), dtype=u_node.dtype).T
        for it in range(nt-1):
            dt = t[it+1] - t[it]
            df0 = np.asarray(app.df(u_node[it]), dtype=u_node.dtype)
            f0 = np.asarray(app.f(u_node[it]), dtype=u_node.dtype)
            f1 = np.asarray(app.f(u_node[it+1]), dtype=u_node.dtype)
            fm = np.asarray(app.f(0.5*(u_node[it]+u_node[it+1])), dtype=u_node.dtype)
            r1 = f1 - f0 - df0@(u_node[it+1]-u_node[it])
            rm = fm - f0 - 0.5*df0@(u_node[it+1]-u_node[it])
            estnl[it] = dt * (1 / 6) * np.sum(r1 ** 2)
            estnl[it] += dt * (2 / 3) * np.sum(rm ** 2)
            du = u_node[it+1]-u_node[it]
            r = df0@du
            # print(f"{it=} {u1[it]=} {np.sum(du*du)=} {np.sum(r*r)=}")
            # alpha = self.alpha*np.fmin(dt,1)
            alpha=0
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
