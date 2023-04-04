import numpy as np


#------------------------------------------------------------------
def compute_error_sequence(app, method, ref='random', itermax=10, nmax = None, add_errors=[], functionals=[]):
    import utils
    errors_plot = [method.error_type]
    errors_plot.extend(add_errors)
    data_plots = {'refine': ref, 't':[], 'u_mid':[], 'u_node':[], 'err_fct':[], 'est':[], 'duals':{}}
    data_per_iter = {'n': [], 'err': [], 'quot':[[method.error_type,'eta']]}
    print(f"{data_per_iter['quot']=}")
    ns = []
    for iter in range(itermax):
        if ref == 'random' or ref == 'uniform':
            n = 10*2**iter
            t = np.linspace(0, app.T, n)
            if ref == 'random': t[1:-1] += (t[2:] - t[:-2])*(np.random.rand(n-2)-0.5)*0.375
        elif ref == 'adaptive':
            if iter:
                t, info = utils.adapt_mesh(t, est['nl'] + est['ap'])
            else:
                t = np.linspace(0,app.T, 10)
            n = len(t)
        else:
            raise ValueError(f"unknown refinement {ref}")
        if nmax is not None and n > nmax: break
        ns.append(n)
        # tm = 0.5*(t[:-1]+t[1:])
        u_ap = method.run_forward(t, app)
        errfct, err = method.compute_error(t, app.sol_ex, app.dsol_ex, u_ap)
        est, estval = method.estimator(t, u_ap, app)
        u_node, u_mid = method.interpolate(t, u_ap)
        data_plots['t'].append(t)
        data_plots['u_node'].append(u_node)
        data_plots['u_mid'].append(u_mid)
        # data_plots['uex'].append(app.sol_ex(t))
        if iter==0:
            erkeys = err.keys() if errors_plot=="all" else set(errors_plot).intersection(set(err.keys()))
            e = {k:[] for k in erkeys}
            for k in est.keys(): e[k] = []
            e['eta'] = []
        for k in erkeys:
           e[k].append(np.sqrt(np.sum(err[k]**2)))
        e['eta'].append(estval['sum'])
        for k in est.keys():
            e[k].append(np.sqrt(np.sum(est[k])))
        if method.error_type == "L2":
            data_plots['err_fct'].append(np.sum(errfct['err_node'],axis=1))
        elif method.error_type=="H1":
            data_plots['err_fct'].append(np.sum(errfct['err_der'],axis=1))
        else:
            raise ValueError(f"unknown {method.error_type=}")
        data_plots['est'].append(np.sqrt( np.sum(np.array([est[k] for k in est.keys()]),axis=0) ))
        for i,functional in enumerate(functionals):
            J_ap = method.compute_functional(t, u_ap, functional)
            J = method.compute_functional(t, app.sol_ex, functional)
            z_ap = method.run_backward(t, u_ap, app, functional)
            z_node, z_mid = method.interpolate_dual(t, z_ap)
            est_d, estval_d = method.estimator_dual(t, u_ap, z_ap, app, functional)
            J_dual = method.compute_functional_dual(t, z_ap, app)
            add = f"_{i:1d}"
            if iter==0:
                e["J"+add] = []
                for k in est_d.keys(): e["d"+add+k] = []
                e["eta_d"+add] = []
                e["eta_J"+add] = []
                data_plots["z"+add] = []
                data_plots["duals"]["z"+add] = "z"+functional.name
                data_per_iter['quot'].append(["J"+add,"eta_J"+add])
            e["J"+add].append(np.fabs(J_ap-J))
            for k in est_d.keys():
                e["d"+add+k].append(np.sqrt(np.sum(est_d[k])))
            e["eta_d"+add].append(estval_d['sum'])
            e["eta_J"+add].append(estval['sum'] * estval_d['sum'])
            data_plots["z" + add].append(z_node)

    # for k in e.keys(): e[k] = np.array(e[k])
    data_per_iter['n'] = np.array(ns)
    data_per_iter['err'] = {k:np.array(e[k]) for k in e.keys()}
    return data_per_iter, data_plots

# ------------------------------------------------------------------
def plot_error_sequence(app, method, data, kplot):
    import matplotlib.pyplot as plt
    if len(data["duals"]): nrows=4
    else: nrows=3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True)
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
        axs[0].plot(t, uex[j], '--', label=f'exact({j:1d})')
        axs[0].plot(t, un[:,j], '-.', label=f'u_node({j:1d})')
        if len(tm)<100:
            axs[0].plot(tm, um[:,j], 'D', label=f'u_mid({j:1d})')
    if method.error_type=="L2": lab = r'$\|e\|$'
    elif method.error_type == "H1": lab = r'$\|e_t\|$'
    else: raise ValueError(f"unknown {method.error_type=}")
    axs[1].plot(tm, errfct, label=lab)
    axs[1].plot(tm, est, label=f'est')
    axs[0].set_title(f"solutions")
    axs[1].set_title(f"error {method.error_type}")
    axs[2].plot(tm, t[1:]-t[:-1], label=r'$_{\delta}$')
    for k,v in data["duals"].items():
        axs[3].set_title(f"duals")
        axs[3].plot(t, data[k][kplot], label=v)
    for ax in axs:
        ax.legend()
        ax.grid()
    plt.show()

# ------------------------------------------------------------------
def print_error_sequence(app, method, data):
    ns, e, quot = data['n'], data['err'], data['quot']
    print("\n"+(20+19*len(e.keys()))*"_")
    print(f"{method.name:^20s} {app.name:^20s}")
    print((20+19*len(e.keys())+10*len(quot))*"_")
    s = f"{'n':^10s}"
    for k in e.keys(): s += f"{k:^19s}"
    for q in quot:
        s += f"{q[0]:>4s}/{q[1]:<4s}"
    print(s)
    print((20+19*len(e.keys()))*"_")
    for i,n in enumerate(ns):
        s = f"{n:10d}"
        if i:
            for k in e.keys():
                s += f"{e[k][i]:11.2e} ({e[k][i-1]/e[k][i]:5.2f})"
        else:
            for k in e.keys(): s += f"{e[k][i]:11.2e} (-----)"
        for q in quot:
            fq = float( e[q[0]][i]/e[q[1]][i])
            s += f"{fq:10.2f}"
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
def test_errors(method, app, add_errors =[], itermax=10, ref='random', functionals=[]):
    kplots = np.arange(0,itermax,5)
    data_print, data_plot = compute_error_sequence(app, method, itermax=itermax, add_errors=add_errors, ref=ref, functionals=functionals)
    for kplot in kplots:
        plot_error_sequence(app, method, data_plot, kplot)
    print_error_sequence(app, method, data_print)
    import matplotlib.pyplot as plt
    plt.title(f"rates {app.name} {method.name}")
    plt.loglog(data_print['n'], data_print['err'][method.error_type], label=f'e ({method.error_type})')
    plt.loglog(data_print['n'], data_print['err']['eta'], '-X', label=f'est')
    plt.legend()
    plt.grid()
    plt.show()

#------------------------------------------------------------------
if __name__ == "__main__":
    import applications
    from SIMOS_GM.ode.applications import analytical_solutions
    import cgp
    # method = cg2.Cg2P()
    # method = cg1.Cg1D(alpha=0.1)
    # method = cg1.Cg1P(alpha=0)
    method = cgp.CgP(k=2)
    app = analytical_solutions.Exponential()
    app = analytical_solutions.Quadratic()
    app = applications.BockTest()
    app = analytical_solutions.Oscillator()
    app = analytical_solutions.Stiff1(a=100)
    # app = analytical_solutions.QuadraticIntegration()
    # app = analytical_solutions.LinearIntegration()
    # app = analytical_solutions.SinusIntegration()


    # add_errors = []
    add_errors = ['L2']
    import classes
    functionals = [classes.FunctionalEndTime(0), classes.FunctionalMean(0)]
    functionals = []
    test_errors(method, app, itermax=10, functionals=functionals, add_errors=add_errors)

    # test_adaptive(method, apps, itermax=10)


