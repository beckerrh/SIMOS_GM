from matplotlib import ticker
import numpy as np

#------------------------------------------------------------------
def plot_sol_mesh_est(t, app, u_node, est, kiter, z_node=None, est_d=None):
    nplots = 3
    if z_node is not None: nplots +=1
    # fig = plt.figure(figsize=plt.figaspect(1/nplots))
    fig = plt.figure(figsize=1.5*plt.figaspect(1))
    fig.suptitle(f"{kiter=}  n={len(t)}")
    figshape = (1, nplots)
    figshape = (2, 2)
    app.plot(fig, t, u_node, axkey=(*figshape, 1), label_ad="u")
    ax = fig.add_subplot(*figshape, 4)
    tm = 0.5 * (t[1:] + t[:-1])
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
    ax.yaxis.tick_right()
    plt.plot(tm, t[1:] - t[:-1], '-X', label=r'$\delta$')
    plt.legend()
    plt.grid()
    ax = fig.add_subplot(*figshape, 3)
    for key in est.keys():
        plt.plot(tm, est[key], label=f"p {key}")
        if z_node is not None:
            plt.plot(tm, est_d[key], label=f"d {key}")
    plt.legend()
    plt.grid()
    if z_node is not None:
        app.plot(fig, t, z_node, axkey=(*figshape, 2), label_ad="z")
    plt.show()
