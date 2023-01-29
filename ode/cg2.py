import numpy as np

def cg2_si(u, f, df, t):
    u = np.asarray(u)
    print(f"{u.shape=}")
    print(f"{t=}")
    dim, nt = u.shape[0], t.shape[0]
    u1 = np.empty(shape=(nt, dim))
    u2 = np.empty(shape=(nt - 1, dim))
    u1[0] = u
    b = np.empty(2*dim)
    A = np.empty((2*dim,2*dim))
    M = np.eye(dim)
    for it in range(nt-1):
        dt = t[it+1]-t[it]
        f0 = np.asarray(f(u1[it]))
        A0 = np.asarray(df(u1[it]))
        print(f"{f0=} {A0=} {u1[it]=}")
        f0 -= A0 @ u1[it]
        b[:dim] = dt*f0
        b[dim:] = 0.5*dt*f0
        b[:dim] += (M+(dt/2)*A0)@u1[it]
        b[dim:] += (0.5*M+(dt/3)*A0)@u1[it]
        A[:dim, :dim] = M - (dt/2)*A0
        A[:dim, dim:] = (dt/6)*A0
        A[dim:, :dim] = 0.5*M - (dt/6)*A0
        A[dim:, dim:] = (1/6)*M - (dt/12)*A0
        usol = np.linalg.solve(A, b)
        u1[it + 1] = usol[:dim]
        u2[it] = usol[dim:]
    return u1, u2

#------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f = lambda u: [0.2*u]
    df = lambda u: [0.2]

    t = np.linspace(0, 10, 20)
    u1, u2 = cg2_si([1], f, df, t)
    plt.plot(t, u1, 'x', label='p1')
    plt.plot(t, np.exp(0.2*t), label='exact')
    plt.legend()
    plt.grid()
    plt.show()
