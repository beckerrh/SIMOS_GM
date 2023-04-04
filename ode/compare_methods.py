import numpy as np

#------------------------------------------------------------------
def compare_methods(app, methods, n):
    import matplotlib.pyplot as plt
    u = {}
    # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False)
    nm = len(methods)
    fig = plt.figure(figsize=plt.figaspect(0.75))
    fig.suptitle(f"compare_methods app={app.name}")
    pltcount = 1
    for method in methods:
        t = np.linspace(0, app.T, n)
        tm = 0.5 * (t[1:] + t[:-1])
        sol_ap = method.run_forward(t, app)
        est, estval = method.estimator(t, sol_ap, app)
        u_ap, u_apm = method.interpolate(t, sol_ap)
        app.plot(fig, t, u_ap, axkey=(nm, 2, pltcount), title="u"+method.name)
        pltcount += 1
        ax = fig.add_subplot(nm, 2, pltcount)
        ax.set_title(f'{method.name}')
        pltcount += 1
        for k in est.keys():
            ax.plot(tm, est[k], label=f'est_{k}')
        ax.legend()
        ax.grid()
    plt.show()
    return u

