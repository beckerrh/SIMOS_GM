import numpy as np
import matplotlib.pyplot as plt
import utils, plotutils

#------------------------------------------------------------------
def run_adaptive(app, method, n0, itermax, nplots=5, eps=1e-4, theta=0.8, F=None):
    kplots = [(i+1) * (itermax - 1) // nplots for i in range(nplots)]
    ns, estvals_p, estvals_d = [], [], []
    for kiter in range(itermax):
        if kiter:
            eta_p = est_p['nl'] + est_p['ap']
            if F is not None:
                eta_d = est_d['nl'] + est_d['ap']
                eta = (np.sum(eta_p)*eta_d + np.sum(eta_d)*eta_p)/(np.sum(eta_d)+np.sum(eta_p))
                eta += eta_p
            else:
                eta = eta_p
            t, refinfo = utils.adapt_mesh(t, eta, theta=theta)
        else:
            t = np.linspace(0, app.T, n0)
            refinfo = None
        u_ap = method.run_forward(t, app)
        est_p, estval_p = method.estimator(t, u_ap, app)
        if F is not None:
            uT_ap = method.compute_functional(t, u_ap, F)
            z_ap = method.run_backward(t, u_ap, app, F)
            est_d, estval_d = method.estimator_dual(t, u_ap, z_ap, app, F)
            # print(f"{estval_d=}")
            # axs[1].plot(tm, z_ap[1], label=f"z_{k} EP")
            # axs[1].plot(np.repeat(t[-1],2), z_ap[2], 'X', label=f"z_{k} EP")
            uT_apd = method.compute_functional_dual(t, z_ap, app)
            est_goafem = estval_p['sum']*estval_d['sum']+estval_p['sum']**2
            estvals_d.append(estval_d['sum'])
            # print(f"{uT_ap=:9.2e}  err_val={np.fabs(val-uT_ap):8.2e} {est_goafem=:8.2e}")
            z_node, z_mid = method.interpolate_dual(t, z_ap)
        else:
            z_node = None
        if kiter:
            for key in est_p.keys(): ests_p[key].append(np.sqrt(np.sum(est_p[key])))
            if F is not None:
                for key in est_d.keys(): ests_d[key].append(np.sqrt(np.sum(est_d[key])))
        else:
            ests_p = {key:[np.sqrt(np.sum(est_p[key]))] for key in est_p.keys()}
            if F is not None:
                ests_d = {key: [np.sqrt(np.sum(est_d[key]))] for key in est_d.keys()}
            else:
                est_d = None
                z_node = None
        u_node, u_mid = method.interpolate(t, u_ap)
        print(f"{u_node[-1]=}")
        estvals_p.append(estval_p['sum'])
        ns.append(len(t))
        if kiter in kplots:
            plotutils.plot_sol_mesh_est(t, app, u_node, est_p, kiter, z_node, est_d)
        rho_p = estvals_p[-1]/estvals_p[max(-2,-kiter)]
        if F is not None:
            rho_d = estvals_d[-1]/estvals_d[max(-2,-kiter)]
            eta = estvals_p[-1] * estvals_d[-1]
        else:
            rho_d=1
            eta = estvals_p[-1]
            uT_ap = uT_apd = 0
        nperc = 0
        if refinfo: nperc=refinfo['nperc']
        if kiter:
            message = f"{kiter:3d} {len(t):8d} ({nperc:4.2f}) {eta:7.2e} "
            message += f"{rho_p:5.2f} {rho_d:5.2f} {uT_ap:13.8e}"
        else:
            message = 68*'='+'\n'
            message += f"{'iter':3s} {'n (%ref)':>15s} {'eta':>7} {'rho_p':>5s} {'rho_d':>5s} {'F_ap':>13s}"
            message += '\n' + 68 * '='
        print(message)
        if eta < eps:
            plotutils.plot_sol_mesh_est(t, app, u_node, est_p, kiter, z_node, est_d)
            print(f"Tolerance achieved {eta:7.2e}<{eps:7.2e}")
            break
    fig = plt.figure(figsize=plt.figaspect(1))
    fig.suptitle(f"rates")
    estvals_p = np.array(estvals_p)
    plt.loglog(ns, estvals_p, 'x-', label=r'$\eta_p$')
    if F is not None:
        estvals_d = np.array(estvals_d)
        plt.loglog(ns, estvals_d, 'x-', label=r'$\eta_d$')
        plt.loglog(ns, estvals_p*estvals_d, 'x-', label=r'$\eta$')
    # for key in ests_p.keys():
    #     plt.loglog(ns, ests_p[key], label=f"{key} (p)")
    # for key in ests_d.keys():
    #     plt.loglog(ns, ests_d[key], label=f"{key} (d)")
    plt.legend()
    plt.grid()
    plt.show()


