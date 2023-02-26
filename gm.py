import numpy as np
from SIMOS_GM.utility import algodata

#------------------------------------------------------------------
def _gm_fix(x, f, gradf, eps, t, maxiter, history, verbose):
    if history: xhist = [x]
    for iter in range(maxiter):
        g = gradf(x)
        gn2 = np.dot(g,g)
        # if iter==0: eps *= gn2
        if verbose: print(f"{iter:4d} {gn2:12.3e}")
        if gn2 <= eps:
            if history: return np.array(xhist)
            return x
        x -= t*g
        if history: xhist.append(x)
    raise ValueError(f"{gm.__name__}: too many iterations {maxiter=}")

#------------------------------------------------------------------
def _gm_bt(x, f, gradf, eps, t, maxiter, nbt, omegabt, alphabt, history, verbose):
    if history: xhist = [x]
    ibt=-1
    for iter in range(maxiter):
        g = gradf(x)
        gn2 = np.dot(g,g)
        # if iter==0: eps *= gn2
        if verbose: print(f"{iter:4d} {gn2:12.3e} {t:12.3e} ({ibt:3d})")
        if gn2 <= eps:
            if history: return np.array(xhist)
            return x
        y = np.copy(x)
        fy = f(y)
        for ibt in range(nbt):
            x = y - t*g
            if f(x) <= fy - alphabt*t*gn2: break
            t *= omegabt
        if ibt == nbt-1: raise ValueError(f"{gm.__name__}: too many inner iterations {nbt=}")
        t /= omegabt
        if history: xhist.append(x)
    raise ValueError(f"{gm.__name__}: too many iterations {maxiter=}")

#------------------------------------------------------------------
def _gm_fix_stopcrit(x, f, gradf, eps, t, mu, maxiter, history, verbose):
    if history: xhist = [x]
    q = np.finfo(x.dtype).min
    v = np.copy(x)
    lam = -1
    for iter in range(maxiter):
        g = gradf(x)
        gn2 = np.dot(g,g)
        if verbose: print(f"{iter:4d} {gn2:10.3e} {lam:10.3e}")
        fx = f(x)
        if fx <= q+eps:
            if history: return np.array(xhist)
            return x
        w = x - g/mu
        p = fx - gn2/mu/2
        Q = np.dot(w-v,w-v)*mu/2
        print(f"{p-q=:10.3e} {(p-q)/Q/2=:10.3e}")
        lam = 0.5 + (p-q)/Q/2
        if lam < 0 : lam = 0
        if lam > 1 : lam = 1
        q = (1-lam)*q + lam*p + lam*(1-lam)*Q
        v = (1-lam)*v + lam*w
        # print(f"\t{lam=} {Q=} {fx-q=} {gn2/mu/2=}")
        x -= t*g
        if history: xhist.append(x)
    raise ValueError(f"{gm.__name__}: too many iterations {maxiter=}")

#------------------------------------------------------------------
def gm(x, f, gradf, datain=algodata.AlgoInput()):
    if not hasattr(datain,'mu'): raise ValueError(f"{gm.__name__} needs 'mu'")
    if hasattr(datain,'L'):
        # return _gm_fix(x, f, gradf, datain.mu*datain.tol, 1/datain.L,
        #                datain.maxiter, datain.history, datain.verbose)
        return _gm_fix_stopcrit(x, f, gradf, datain.tol, 1 / datain.L, datain.mu,
                       datain.maxiter, datain.history, datain.verbose)
    else:
        return _gm_bt(x, f, gradf, datain.mu * datain.tol, datain.t,
                       datain.maxiter, datain.nbt, datain.omegabt, datain.alphabt,
                       datain.history, datain.verbose)


#------------------------------------------------------------------
def test1d():
    import matplotlib.pyplot as plt
    f = lambda x: (x-1)**2 + np.exp(x) + np.sin(10*x)
    gf = lambda x: 2*(x-1)+ np.exp(x) + 10*np.cos(10*x)
    datain = algodata.AlgoInput()
    # datain.L = 500
    datain.t = 1/500
    datain.mu = 2
    datain.history = True
    xhist = gm(4, f, gf, datain)
    plt.plot(xhist, f(xhist), 'x', label='path')
    x = np.linspace(-2,4)
    plt.plot(x, f(x), label='f')
    plt.legend()
    plt.grid()
    plt.show()

#------------------------------------------------------------------
def testls():
    import matplotlib.pyplot as plt
    import examples
    dim = 20
    ls = examples.least_squares.LeastSquares(dim=dim, m=2*dim)
    datain = algodata.AlgoInput()
    datain.history = True
    datain.t = 0.01
    datain.mu = ls.mu
    datain.L = ls.L
    datain.maxiter = 1000
    xhist = gm(np.zeros(dim), ls.f, ls.grad, datain)


#------------------------------------------------------------------
if __name__ == "__main__":
    # test1d()
    testls()
