import numpy as np
from SIMOS_GM.utility import algodata

#------------------------------------------------------------------
def minres(x, res, dres, datain=algodata.AlgoInput(), verbose=True):
    f = lambda x: 0.5*np.sum(res(x)**2)
    mu = datain.mu
    t, nbt, omegabt, alphabt = datain.t, datain.nbt, datain.omegabt, datain.alphabt
    ibt, t = -1, datain.t
    for iter in range(datain.maxiter):
        r = res(x)
        dr = dres(x)
        m,n = dr.shape
        U,S,VT = np.linalg.svd(dr.T)
        # print(f"{r.shape=} {dr.shape=} {S.shape=} {U.shape=} {VT.shape=} {np.diag(1/(S+mu)).shape=}")
        # print(f"{x=} {r=} {dr=}")
        dx = U@np.diag(S/(S**2+mu))@VT[:n,:]@r
        dn2 = np.dot(dx,dx)
        # dn2 = np.sum((dr.T@r)**2)
        fx = f(x)
        if verbose: print(f"{iter:4d} {fx:11.3e} {dn2:11.3e} ({ibt:3d} {t:9.2e})")
        if dn2 <= datain.tol:
            return x
        y = np.copy(x)
        for ibt in range(nbt):
            x = y - t*dx
            # print(f"{f(x)=} {fx=} {dn2=} {t=}")
            if f(x) <= fx - alphabt*t*dn2: break
            t *= omegabt
        if ibt == nbt-1: raise ValueError(f"{minres.__name__}: too many inner iterations {nbt=} {t=} {dn2=}")
        t /= omegabt
    raise ValueError(f"no convergence after {datain.maxiter} iterations")
