import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#------------------------------------------------------------------
def compare(apps, methods, n):
    def _run_single(app, method, t, outer1, outer2, title):
        sol_ap = method.run_forward(t, app)
        est, estval = method.estimator(t, sol_ap, app)
        u_ap = method.interpolate(t, sol_ap)
        app.change_solution(u_ap)
        app.plot(fig=fig, gs=outer1, t=t, u=u_ap, title=title)
        ax = fig.add_subplot(outer2)
        tm = 0.5 * (t[1:] + t[:-1])
        for k in est.keys():
            ax.plot(tm, est[k], label=f'est_{k}')
        ax.legend()
        ax.grid()
    if isinstance(methods, list):
        assert not isinstance(apps, list)
        nplots, T, name, ncols, runmethods = apps.nplots, apps.T, apps.name, len(methods), True
    else:
        assert not isinstance(methods, list)
        nplots, T, name, ncols, runmethods = apps[0].nplots, apps[0].T, methods.name, len(apps), False
    fig = plt.figure(figsize=plt.figaspect(nplots*0.5))
    fig.suptitle(f"compare app={name}")
    outer = gridspec.GridSpec(nplots+1, ncols, wspace=0.4, hspace=0.4)
    t = np.linspace(0, T, n)

    if runmethods:
        for i,method in enumerate(methods): _run_single(apps, method, t, outer[:-1, i], outer[-1, i], title=method.name)
    else:
        for i, app in enumerate(apps): _run_single(app, methods, t, outer[:-1, i], outer[-1, i], title=app.name)
    plt.show()

