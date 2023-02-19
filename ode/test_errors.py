import numpy as np


#------------------------------------------------------------------
def compute_error_sequence(app, method, ref='random', kmax=10, nmax = None, add_errors=[]):
    import utils
    errors_plot = [method.error_type]
    errors_plot.extend(add_errors)
    # if plotting == True and kplots is None: kplots=[kmax//5]
    data_plots = {'refine': ref, 't':[], 'u_mid':[], 'u_node':[], 'err_fct':[], 'est':[]}
    ns = []
    for kiter in range(kmax):
        if ref == 'random' or ref == 'uniform':
            n = 10*2**kiter
            t = np.linspace(0, app.T, n)
            if ref == 'random': t[1:-1] += (t[2:] - t[:-2])*(np.random.rand(n-2)-0.5)*0.375
        elif ref == 'adaptive':
            if kiter:
                t, info = utils.adapt_mesh(t, est['nl'] + est['ap'])
            else:
                t = np.linspace(0,app.T, 10)
            n = len(t)
        else:
            raise ValueError(f"unknown refinement {ref}")
        if nmax is not None and n > nmax: break
        tm = 0.5*(t[:-1]+t[1:])
        sol_ap = method.run_forward(t, app)
        errfct, err = method.compute_error(t, app.sol_ex, app.dsol_ex, sol_ap)
        est, estval = method.estimator(t, sol_ap, app)
        u_node, u_mid = method.interpolate(t, sol_ap)
        data_plots['t'].append(t)
        data_plots['u_node'].append(u_node)
        data_plots['u_mid'].append(u_mid)
        # data_plots['uex'].append(app.sol_ex(t))
        if not len(ns):
            erkeys = err.keys() if errors_plot=="all" else set(errors_plot).intersection(set(err.keys()))
            e = {k:[] for k in erkeys}
            for k in est.keys():
                e[k] = []
            estvals = []
        for k in erkeys:
           e[k].append(np.sqrt(np.sum(err[k]**2)))
        estvals.append(estval['sum'])
        for k in est.keys():
            e[k].append(np.sqrt(np.sum(est[k])))
        ns.append(n)
        if method.error_type == "L2":
            data_plots['err_fct'].append(np.sum(errfct['err_node'],axis=1))
        elif method.error_type=="H1":
            data_plots['err_fct'].append(np.sum(errfct['err_der'],axis=1))
        else:
            raise ValueError(f"unknown {method.error_type=}")
        estall = np.zeros_like(tm)
        for k in est.keys():
            estall += est[k]
        data_plots['est'].append(np.sqrt(estall))
    for k in e.keys(): e[k] = np.array(e[k])
    data_per_iter = {'n': np.array(ns), 'err': e, 'est': estvals}
    return data_per_iter, data_plots

# ------------------------------------------------------------------
def plot_error_sequence(app, method, data, kplot):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False)
    fig.suptitle(f"{method.name} {app.name} (iter={kplot})")
    t = data['t'][kplot]
    un = data['u_node'][kplot]
    um = data['u_mid'][kplot]
    errfct = data['err_fct'][kplot]
    est = data['est'][kplot]
    uex = app.sol_ex(t)
    dim = un.shape[1]
    tm = 0.5*(t[:-1]+t[1:])
    for j in range(dim):
        axs[0].plot(t, un[:,j], 'x', label=f'u_node({j:1d})')
        axs[0].plot(t, uex[j], label=f'exact({j:1d})')
        axs[0].plot(tm, um[:,j], 'D', label=f'u_mid({j:1d})')
    if method.error_type=="L2": lab = r'$\|e\|$'
    elif method.error_type == "H1": lab = r'$\|e_t\|$'
    else: raise ValueError(f"unknown {method.error_type=}")
    axs[1].plot(tm, errfct, label=lab)
    axs[1].plot(tm, est, label=f'est')
    axs[0].set_title(f"solutions")
    axs[1].set_title(f"error {method.error_type}")
    axs[2].plot(tm, t[1:]-t[:-1], label=r'$_{\delta}$')
    for ax in axs:
        ax.legend()
        ax.grid()
    plt.show()

# ------------------------------------------------------------------
def print_error_sequence(app, method, data):
    ns, e, est = data['n'], data['err'], data['est']
    import matplotlib.pyplot as plt
    print("\n"+(20+19*len(e.keys()))*"_")
    print(f"{method.name:^20s} {app.name:^20s}")
    print((20+19*len(e.keys()))*"_")
    s = f"{'n':^10s}"
    for k in e.keys(): s += f"{k:^19s}"
    s += f"{'err/est':10s}"
    print(s)
    print((20+19*len(e.keys()))*"_")
    for i,n in enumerate(ns):
        if i:
            s = f"{n:10d}"
            for k in e.keys(): s += f"{e[k][i]:11.2e} ({e[k][i-1]/e[k][i]:5.2f})"
        else:
            s = f"{n:10d}"
            for k in e.keys(): s += f"{e[k][i]:11.2e} (-----)"
        quot = float( e[method.error_type][i]/est[i])
        s += f"{quot:10.2f}"
        print(s)

# ------------------------------------------------------------------
def test_adaptive(method, apps, kmax=30):
    import matplotlib.pyplot as plt
    from mpltools import annotation
    for app in apps:
        data_ada = compute_error_sequence(app, method, ref='adaptive', kmax=kmax)
        nmax = 2*data_ada[0]['n'][-1]
        data_uni = compute_error_sequence(app, method, ref='uniform', nmax=nmax)
        fig, axs = plt.subplots(ncols=1, nrows=2)
        ns = data_ada[0]['n']
        axs[0].loglog(ns, data_ada[0]['err'][method.error_type], label=f'e ({method.error_type} adapt)')
        axs[0].loglog(ns, data_ada[0]['est'], label=f'est (adapt)')
        ns = data_uni[0]['n']
        axs[0].loglog(ns, data_uni[0]['err'][method.error_type], label=f'e ({method.error_type} uni)')
        p = axs[0].loglog(ns, data_uni[0]['est'], label=f'est (uni)')
        axs[0].legend()
        axs[0].grid()
        axs[0].set_title(f"rates {app.name} {method.name}")


        data = data_uni[0]['est']
        slope, intercept = np.polyfit(np.log(ns), np.log(data), 1)
        annotation.slope_marker((ns[0], data[len(data)//2]), np.round(slope,2), ax=axs[0],
                                poly_kwargs={'fill':False, 'edgecolor':p[0].get_color()})

        t = data_ada[1]['t'][-1]
        tm = 0.5*(t[1:]+t[:-1])
        dt = t[1:]-t[:-1]
        axs[1].plot(tm, dt, label=r'$\delta$ (ada)')
        t = data_uni[1]['t'][-1]
        tm = 0.5*(t[1:]+t[:-1])
        dt = t[1:]-t[:-1]
        axs[1].plot(tm, dt, label=r'$\delta$ (uni)')
        axs[1].legend()
        axs[1].grid()
        plt.show()


# ------------------------------------------------------------------
def test_errors(method, apps, add_errors =[], kmax=10, ref='random'):
    kplots = np.arange(0,kmax,5)
    for app in apps:
        data_print, data_plot = compute_error_sequence(app, method, kmax=kmax, add_errors=add_errors, ref=ref)
        for kplot in kplots:
            plot_error_sequence(app, method, data_plot, kplot)
        print_error_sequence(app, method, data_print)
        import matplotlib.pyplot as plt
        plt.loglog(data_print['n'], data_print['err'][method.error_type], label=f'e ({method.error_type})')
        plt.loglog(data_print['n'], data_print['est'], '-X', label=f'est')
        plt.legend()
        plt.grid()
        plt.title(f"rates {app.name} {method.name}")
        plt.show()

#------------------------------------------------------------------
if __name__ == "__main__":
    import analytical_solutions
    import cg1, cg2, cgp
    # method = cg2.Cg2P()
    # method = cg1.Cg1D(alpha=0.1)
    # method = cg1.Cg1P(alpha=0)
    method = cgp.CgP(k=2)
    apps = [analytical_solutions.Exponential()]
    apps = [analytical_solutions.Quadratic()]
    apps = [analytical_solutions.Oscillator()]
    apps = [analytical_solutions.QuadraticIntegration()]
    apps = [analytical_solutions.SinusIntegration()]


    # add_errors = []
    test_errors(method, apps, add_errors = ['L2_mid', 'L2_nod', 'L2', 'max_mid'], kmax=5)

    # test_adaptive(method, apps, kmax=10)


