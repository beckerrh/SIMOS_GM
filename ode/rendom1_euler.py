import numpy as np  
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print("***", SCRIPT_DIR)
from SIMOS_GM.ode import classes
# import classes

#==================================================================
class CgP(classes.Method):
#==================================================================
    """
    cgp on [-1,1]
    """
    def __init__(self, k=1, plotbasis=False, alpha=0):
        # alpha=0: CN alpha=1: Euler
        self.alpha = alpha
        super().__init__(error_type = "H1")
        self.name = self.name + f"_{k}"
        assert k>=1
        self.k = k
        self.int_x, self.int_w = np.polynomial.legendre.leggauss(k+1)
        # self.int_x, self.int_w = np.polynomial.legendre.leggauss(k+2)
        self.int_n = len(self.int_w)
        self.psi, self.phi = [], []
        for i in range(k):
            p = np.polynomial.legendre.Legendre.basis(deg=i, domain=[-1, 1])
            self.psi.append(p)
            q = p.integ()
            self.phi.append(q-q(-1))
            assert self.phi[-1].deriv() == self.psi[-1]
        if plotbasis:
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
                plt.plot(t, self.phi[i](t), '-', label=r"$\phi_{:2d}$".format(i))
            plt.legend()
            plt.grid()
            plt.show()
        self.M = np.zeros(shape=(k,k))
        self.T = np.zeros(shape=(k,k))
        self.int_phiphi = np.zeros(shape=(k,k))
        self.int_psipsi = np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                for ii in range(self.int_n):
                    self.M[i,j] += self.int_w[ii]* self.psi[i](self.int_x[ii])* self.phi[j](self.int_x[ii])
                    self.T[i, j] += self.int_w[ii] * self.psi[i](self.int_x[ii]) * self.psi[j](self.int_x[ii])
                    self.int_phiphi[i, j] += self.int_w[ii] * self.phi[i](self.int_x[ii]) * self.phi[j](self.int_x[ii])
                    self.int_psipsi[i, j] += self.int_w[ii] * self.psi[i](self.int_x[ii]) * self.psi[j](self.int_x[ii])
        self.diagT = np.diagonal(self.T)[1:]
        self.coefM = np.diagonal(self.M, offset=-1)
        if plotbasis:
            i = np.arange(1,k)
            assert np.allclose(self.diagT, 2/(2*i+1))
            assert np.allclose(self.coefM, 2/(4*i**2-1))
            # print(f"{self.diagT=}")
            # print(2/(2*i+1))
            # print(f"{self.coefM=}")
            # print(2/(4*i**2-1))
            sys.exit(1)
        self.int_psi_weights = np.empty(shape=(len(self.psi),self.int_n))
        self.int_psi = np.empty(shape=(len(self.psi),self.int_n))
        self.int_psik = np.empty(shape=(self.int_n))
        self.int_phi = np.empty(shape=(len(self.psi),self.int_n))
        for i in range(k):
            for ii in range(self.int_n):
                self.int_psi_weights[i,ii] = self.psi[i](self.int_x[ii])*self.int_w[ii]
                self.int_psi[i, ii] = self.psi[i](self.int_x[ii])
                self.int_phi[i, ii] = self.phi[i](self.int_x[ii])
        p = np.polynomial.legendre.Legendre.basis(deg=k, domain=[-1, 1])
        self.int_psik = p(self.int_x)
        self.int_phik2 = np.sum((self.phi[-1](self.int_x))**2*self.int_w)
        self.psi_mid = np.array([self.psi[ik](0) for ik in range(self.k)])
        self.phi_mid = np.array([self.phi[ik](0) for ik in range(self.k)])
#------------------------------------------------------------------
    def run_forward(self, t, app, linearization=None, lintrandom=None, q=0):
        if linearization is not None:
            utilde_node, utilde_coef = linearization
        if lintrandom is not None:
            self.alpha=0
        u_ic = app.u_zero()
        dim, nt = app.dim, t.shape[0]
        # print(f"{dim=} {u_ic=}")
        # dim, nt = len(u_ic), t.shape[0]
        # assert dim==app.dim
        apphasl = hasattr(app,'l')
        dt = t[1:] - t[:-1]
        if apphasl:
            tm = 0.5*(t[1:]+t[:-1])
            # fl_vec = np.vectorize(app.l, signature="(k,n)->(p,k,n)", otypes=[float])
            # lint = np.asarray(fl_vec(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
            # print(f"{(tm+0.5*self.int_x[:,np.newaxis]*dt).shape=} {lint.shape=}")
        if linearization is not None:
            uint = utilde_node[:-1, np.newaxis] + np.einsum('ikp,kl->ilp', utilde_coef, self.int_phi)
            lintf = np.asarray(app.f(uint.T)).T.reshape(uint.shape)
        u_node = np.empty(shape=(nt, dim), dtype=app.dtype)
        u_coef = np.empty(shape=(nt-1, self.k, dim), dtype=u_node.dtype)
        bloc = np.empty(shape=(self.k,dim), dtype=u_node.dtype)
        Aloc = np.zeros((self.k,self.k, dim, dim), dtype=u_node.dtype)
        if hasattr(app, 'M'):
            M = app.M
        else:
            M = np.eye(dim, dtype=app.dtype)
        u_node[0] = u_ic
        for it in range(nt-1):
            dt = t[it+1]-t[it]
            assert(dt>0)
            bloc.fill(0)
            if linearization is None:
                utilde = u_node[it]
                f0 = np.asarray(app.f(utilde), dtype=u_node.dtype)
                bloc[0] = dt * f0
            else:
                utilde = 0.5*(utilde_node[it]+utilde_node[it+1])
                # f0 = np.asarray(app.f(utilde), dtype=u_node.dtype)
                # print(f"{lintf[it].shape=} {self.int_psi_weights.shape=}")
                bloc += 0.5*dt*np.einsum('jk,lj->lk', lintf[it], self.int_psi_weights)
            # print(f"{utilde.shape=} {utilde=} {app.f(utilde.T)=} {app.f=}")
            A0 = np.asarray(app.df(utilde), dtype=u_node.dtype).reshape(dim, dim)
            # A0 = app.df(utilde)
            if linearization is not None:
                bloc[0] += dt*A0@u_node[it]
            # print(f"{M.shape=} {Aloc[0, 0].shape=} {A0.shape=}")
            # Aloc[0, 0] = 2*M - dt*A0
            Aloc[0, 0] = 2*M - (1+self.alpha)*dt*A0
            for ik in range(1,self.k):
                Aloc[ik, ik] = self.diagT[ik - 1] * M
                Aloc[ik - 1, ik] = 0.5 * dt * self.coefM[ik - 1] * A0
                Aloc[ik, ik - 1] = -0.5 * dt * self.coefM[ik - 1] * A0
            if apphasl:
                bloc += 0.5*dt*np.einsum('jk,lj->lk', lint[it], self.int_psi_weights)
            if lintrandom is not None:
                bloc[0] += lintrandom[it]*utilde
                Aloc[0, 0] += q**2*M*dt
            # print(f"{Aloc.shape=} {Aloc.swapaxes(1,2).shape=} {bloc.shape=}")
            usol = np.linalg.solve(Aloc.swapaxes(1,2).reshape(self.k*dim,self.k*dim), bloc.flat).reshape((self.k,dim))
            # usol = np.linalg.solve(Aloc.swapaxes(1,2).reshape(self.k*dim,self.k*dim), bloc.flat).reshape((self.k,dim))
            # usol = np.linalg.solve(Aloc.reshape(self.k*dim,self.k*dim), bloc.flat).reshape((self.k,dim))
            u_coef[it] = usol
            u_node[it + 1] = 2*usol[0] + u_node[it]
        return u_node, u_coef
#------------------------------------------------------------------
    def run_backward(self, t, u_ap, app, func):
        #app needs to be derivative app!!
        u_node, u_coef = u_ap
        nt, dim = u_node.shape
        funchasl = hasattr(func,'l_prime')
        funchaslT = hasattr(func,'lT_prime')
        dt = t[1:] - t[:-1]
        if funchasl:
            tm = 0.5*(t[1:]+t[:-1])
            uint = u_node[:-1, np.newaxis] + np.einsum('ikp,kl->ilp', u_coef, self.int_phi)
            lint = np.asarray(func.l_prime(tm+0.5*self.int_x[:,np.newaxis]*dt, uint.T)).T
            # print(f"{lint.shape=}")
        z_coef = np.empty(shape=(nt-1, self.k, dim), dtype=u_node.dtype)
        bloc = np.empty(shape=(self.k,dim), dtype=u_node.dtype)
        Aloc = np.zeros((self.k,self.k, dim, dim), dtype=u_node.dtype)
        M = np.eye(dim, dtype=u_node.dtype)
        b0 = np.zeros(shape=(dim), dtype=u_node.dtype)
        b1 = np.zeros_like(b0)
        b2 = np.zeros_like(b0)
        if funchaslT:
            zT = np.asarray(func.lT_prime(u_node[-1]), dtype=u_node.dtype)
        else:
            zT = np.zeros_like(b0)
        for it in range(nt-2,-1,-1):
            dt = t[it+1]-t[it]
            #matrix
            A0 = np.asarray(app.df(0.5*(u_node[it]+u_node[it+1])), dtype=u_node.dtype).reshape(dim, dim).T
            # A0 = np.asarray(app.df(u_node[it]), dtype=u_node.dtype).reshape(dim, dim).T
            for ik in range(1,self.k):
                Aloc[ik, ik] = self.diagT[ik - 1] * M
                Aloc[ik, ik - 1] = 0.5 * dt * self.coefM[ik - 1] * A0
                Aloc[ik - 1, ik] = -0.5 * dt * self.coefM[ik - 1] * A0
            if self.k>1: Aloc[0, 1] *= 0.5
            Aloc[0, 0] = M - 0.5 * dt * A0
        #rhs
            bloc.fill(0)
            if funchasl:
                bloc = 0.5 * dt * np.einsum('jk,lj,j->lk', lint[it], self.int_phi, self.int_w)
                bloc[0] *= 0.5
                # integral (fct test=1)
                b1 = 0.5 * dt * np.einsum('jk,j->k', lint[it], self.int_w)
                b1 -=  bloc[0]
            if it==nt-2:
                # print(f"{bloc[0].shape=} {zT.shape=}")
                bloc[0] += zT
            else:
                bloc[0] += b0
                if self.k > 1: bloc[0] += b2
            z_coef[it] = np.linalg.solve(Aloc.swapaxes(1,2).reshape(self.k*dim,self.k*dim), bloc.flat).reshape((self.k,dim))
            b0 = b1+ (M + 0.5* dt * A0) @ z_coef[it,0]
            if self.k > 1: b2 = -0.25*self.coefM[0]* dt * A0 @ z_coef[it,1]
        return b0, z_coef, zT
# ------------------------------------------------------------------
    def interpolate(self, t, u_ap, mean=False):
        u_node, u_coef = u_ap
        if not mean: return u_node
        return u_node, u_node[:-1]+np.einsum('ijk,j', u_coef, self.phi_mid)
#------------------------------------------------------------------
    def interpolate_dual(self, t, z_ap, mean=False):
        z0, z_coef, zT = z_ap
        z_n = np.empty(shape=(len(t),len(z0)))
        z_n[-1] = np.einsum('kp,k->p', z_coef[-1,:,:], [self.psi[j](1) for j in range(self.k)])
        z_n[1:-1] = 0.5*np.einsum('ikp,k->ip', z_coef[:-1,:,:], [self.psi[j](1) for j in range(self.k)])
        z_n[1:-1] += 0.5*np.einsum('ikp,k->ip', z_coef[1:,:,:], [self.psi[j](-1) for j in range(self.k)])
        z_n[0] = np.einsum('kp,k->p', z_coef[0,:,:], [self.psi[j](-1) for j in range(self.k)])
        # z_n[0] += 0.5*z0
        # z_n[-1] = zT
        # z_n[0] = z0
        if not mean: return z_n
        return z_n, np.einsum('ijk,j', z_coef, self.psi_mid)
#------------------------------------------------------------------
    def compute_functional(self, t, u_ap, func):
        funchasl = hasattr(func, 'l')
        funchaslT = hasattr(func, 'lT')
        assert int(funchasl) + int(funchaslT) == 1
        tm = 0.5 * (t[1:] + t[:-1])
        dt = t[1:] - t[:-1]
        if funchaslT:
            if isinstance(u_ap, tuple):
                u_node, u_coef = u_ap
                return func.lT(u_node[-1])
            else:
                return func.lT(np.array(u_ap(t[-1])).T)
        if isinstance(u_ap, tuple):
            u_node, u_coef = u_ap
            uint = u_node[:-1, np.newaxis] + np.einsum('ikp,kl->ilp', u_coef, self.int_phi)
        else:
            uint = np.array(u_ap(tm + 0.5 * self.int_x[:, np.newaxis] * dt)).T
        lint = func.l(tm + 0.5 * self.int_x[:, np.newaxis] * dt, uint.T).T
        lintval = 0.5*np.einsum('ik,i,k->', lint, dt, self.int_w)
        return lintval
#------------------------------------------------------------------
    def compute_functional_dual(self, t, z_ap, app, linearization=None):
        z0, z_coef, zT = z_ap
        #richtig?
        val = np.dot(app.u_zero(), z0)
        dt = t[1:] - t[:-1]
        if hasattr(app,'l'):
            tm = 0.5 * (t[:-1] + t[1:])
            lint = np.asarray(app.l(tm + 0.5 * self.int_x[:, np.newaxis] * dt)).T
            # print(f"{lint.shape=} {z_coef.shape=} {self.int_psi_weights.shape=} {dt.shape=}")
            val += 0.5*np.einsum('ikp,ilp,lk,i', lint, z_coef, self.int_psi_weights, dt)
        if linearization is not None:
            utilde_node, utilde_coef = linearization
            uint = utilde_node[:-1, np.newaxis] + np.einsum('ikp,kl->ilp', utilde_coef, self.int_phi)
            # uint = np.einsum('ikp,kl->ilp', utilde_coef, self.int_phi)
            # print(f"{uint.shape=} {uint.T.shape=}")
            # f_vec = np.vectorize(app.f, signature="(k,p,n)->(n,p,k)", otypes=[float])
            # lint = np.asarray(f_vec(uint.T)).T
            # print(f"{app.f=}")
            lint = np.asarray(app.f(uint.T)).T.reshape(uint.shape)
            # print(f"{uint.shape=} {lint.shape=} {z_coef.shape=} {self.int_psi_weights.shape=} {dt.shape=}")
            val += 0.5*np.einsum('ikp,ilp,lk,i', lint, z_coef, self.int_psi_weights, dt)
            # val += 0.5 * np.einsum('pki,ilp,lk,i', lint.T, z_coef, self.int_psi_weights, dt)
        return val
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
        dt = t[1:] - t[:-1]
        tm = 0.5 * (t[1:] + t[:-1])
        df0 = np.array([np.sum( (app.df(u_node[it])@u_coef[it,-1])**2) for it in range(nt-1)])
        estap = 0.5 * df0*dt*self.int_phik2
        if apphasl:
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
            lintk = np.einsum('ilk,l,l', lint, self.int_w, self.int_psik)
            estap += 0.5 * np.einsum('ik,ik->i', lintk ** 2, dt[:, np.newaxis])
        df = np.array([np.sum((np.array(app.df(u_node[it])) - np.array(app.df(u_node[it+1]))) ** 2) for it in range(nt - 1)])
        u2 = np.einsum('ikp,ilp,kl->i', u_coef, u_coef, self.int_phiphi)
        estnl = 0.5*dt*df*u2
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
# ------------------------------------------------------------------
    def estimator_dual(self, t, u_ap, z_ap, app, func):
        z0, z_coef, zT = z_ap
        u_node, u_coef = u_ap
        nt, dim = u_node.shape
        dt = t[1:] - t[:-1]
        df = np.array([np.sum((np.array(app.df(u_node[it])) - np.array(app.df(u_node[it+1]))) ** 2) for it in range(nt - 1)])
        z2 = np.einsum('ikp,ilp,kl->i', z_coef, z_coef, self.int_psipsi)
        estnl = (1/96)*dt**3*df*z2
        dfT = np.array([np.array(np.array(app.df(u_node[it])).reshape(dim,dim).T) for it in range(nt - 1)])
        # print(f"{dfT.shape=} {z_coef.shape=} {self.int_psipsi.shape=}")
        z = np.einsum('ipq, ikq, kl->ilp', dfT, z_coef[:,-1:,:], self.int_psi[-1:])
        # print(f"{dt.shape=} {z.shape=} {self.int_w.shape=}")
        estap = (1/96)*dt**3* np.einsum('ilp,ilp,l->i', z,z, self.int_w.shape)
        # jump = np.einsum('ikp,k->ip', z_coef[:-1,:,:], [self.psi[j](1) for j in range(self.k)])
        # jump -= np.einsum('ikp,k->ip', z_coef[1:,:,:], [self.psi[j](-1) for j in range(self.k)])
        # # print(f"{jump.shape=} {(dt[1:]+dt[:-1]).shape=}")
        # estap = np.zeros_like(estnl)
        # jump = np.einsum('ip,ip,i->i', jump, jump, 0.5 * (dt[1:] + dt[:-1]))
        # estap[1:] = 0.5*jump
        # estap[:-1] = 0.5*jump
        # # print(f"{estap=}")
        # jumpT = np.einsum('kp,k->p', z_coef[0,:,:], [self.psi[j](-1) for j in range(self.k)]) -z0
        # estap[0] += np.sum(jumpT*jumpT)*dt[0]
        # print(f"{estap=}")
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
#------------------------------------------------------------------
def test_alpha_wave(method):
    import matplotlib.pyplot as plt
    from SIMOS_GM.ode.applications import analytical_solutions
    app = analytical_solutions.Oscillator()
    t = np.linspace(0,app.T,20)
    u_ap = method.run_forward(t, app)
    u_node, u_mid = method.interpolate(t, u_ap)
    tm = 0.5*(t[1:]+t[:-1])
    fig = plt.figure(figsize=1.5*plt.figaspect(1))
    ax = fig.add_subplot(111)
    ax.plot(t, np.asarray(app.sol_ex(t)).T, 'k--', label='exact')
    ax.plot(tm, u_mid, label=f'cg{method.k} (mid)')
    ax.plot(t, u_node, label=f'cg{method.k} (nod)')
    ax.legend()
    plt.show()
#------------------------------------------------------------------
def test_dual_wave(method, nt=20):
    import matplotlib.pyplot as plt
    from SIMOS_GM.ode.applications import analytical_solutions
    app = analytical_solutions.Oscillator()
    t = np.linspace(0,app.T, nt)
    tm = 0.5*(t[1:]+t[:-1])
    u_ap = method.run_forward(t, app)
    u_node, u_mid = method.interpolate(t, u_ap)
    est_p, estval_p = method.estimator(t, u_ap, app)
    fig = plt.figure(figsize=1.5*plt.figaspect(1))
    ax = fig.add_subplot(511)
    ax.plot(t, np.asarray(app.sol_ex(t)).T, 'k--', label='exact')
    ax.plot(tm, u_mid, label=f'cg{method.k} (mid)')
    ax.plot(t, u_node, label=f'cg{method.k} (nod)')
    ax.legend()

    u_ex = np.asarray(app.sol_ex(t)).T
    functionals = [classes.FunctionalEndTime(0), classes.FunctionalMean(0)]
    for i,functional in enumerate(functionals):
        J_ap = method.compute_functional(t, u_ap, functional)
        J = method.compute_functional(t, u_ex, functional)
        z_ap = method.run_backward(t, u_ap, app, functional)
        z_node, z_mid = method.interpolate_dual(t, z_ap)
        est_d, estval_d = method.estimator_dual(t, u_ap, z_ap, app, functional)
        # print(f"{est_d=}")
        val = method.compute_functional_dual(t, z_ap, app)
        est = estval_p['sum']*estval_d['sum']
        print(f"{J_ap=:10.4e}  err_val={np.fabs(val-J_ap):10.4e} errJ={np.fabs(J_ap-J):10.4e} {est=:8.2e} {estval_p['sum']} {estval_d['sum']}")
        ax = fig.add_subplot(int(f"51{i+2:1d}"))
        ax.set_title(f"{functional.name}")
        ax.plot(t, z_node, label=f"z (mid)")
        ax.plot(tm, z_mid, label=f"z (nod")
        ax.plot(np.repeat(t[-1],2), z_ap[2], 'X', label=f"z")
        ax.plot(np.repeat(t[0],2), z_ap[0], 'X', label=f"z")
        ax.legend()
        ax = fig.add_subplot(int(f"51{i+4:1d}"))
        ax.set_title(f"eta {functional.name}")
        for k,est in est_d.items(): ax.plot(tm, est, '-x', label=f"{k} (dual)")
        for k,est in est_p.items(): ax.plot(tm, est, '-x', label=f"{k} (primal)")
        ax.legend(loc='right')
    plt.show()
#------------------------------------------------------------------
if __name__ == "__main__":
    cgp = CgP(k=5, plotbasis=True)
    # test_alpha_wave(method=cgp)
    # test_dual_wave(method=cgp, nt=20)
