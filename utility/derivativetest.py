import numpy as np

def test_grad(f, gradf, xs, eps=1e-6):
    if isinstance(xs, np.ndarray):
        xs = [xs]
    xm = np.meshgrid(*xs)
    rmax = -np.inf
    ndim = len(xs)
    for x in np.nditer(xm):
        x = np.atleast_1d(x)
        g = gradf(x)
        for i in range(ndim):
            xl = np.copy(x)
            xl[i] -= eps
            xr = np.copy(x)
            xr[i] += eps
            r = abs(f(xr)-f(xl) - 2*eps*g[i])
            rmax = max(r,rmax)
    return rmax
