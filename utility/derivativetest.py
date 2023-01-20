import numpy as np

def test_grad(f, gradf, xs, eps=1e-6):
    rmax = -np.inf
    for x in xs:
        g = gradf(x)
        for i in range(x.shape[0]):
            xl = np.copy(x)
            xl[i] -= eps
            xr = np.copy(x)
            xr[i] += eps
            r = abs(f(xr)-f(xl) - 2*eps*g[i])
            rmax = max(r,rmax)
    return rmax
