import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax0, axs, axl, axr) = plt.subplots(
    ncols=4,
    figsize=(12, 3),
    width_ratios=[1, 3, 1, 1]
)
fig.tight_layout()

#---------------------------------
L = 22
alpha = 0.3
periods = 3
#---------------------------------

def phi(t):
    t = t%L
    if t < (1-alpha)*L:
        return t/(1-alpha), False
    return (L - t)/alpha, True

choice = 0
if choice==1:
    shift = np.arange(L)
    name = 'harfe'
elif choice==2:
    shift = np.ones(L)*alpha*L
    shift[::2] = 0
    name = 'block'
else:
    shift = np.arange(L)*alpha*L
    name = 'opt?'

fig.suptitle(f"{alpha=:4.2f}   sigma={alpha/(1-alpha):4.2f}   maen_back={alpha*L:3.1f}")
axl.set_xlim([0, L])
axl.set_ylim([0, L])
axr.set_xlim([0, L])
axr.set_ylim([0, L])
ax0.set_xlim([0, L])
ax0.set_ylim([0, L])
axs.set_xlim([0, periods*L])
axs.set_ylim([0, L/2+1])

ax0.plot([0, L], [(1-alpha)*L]*2, 'r')
axl.plot([0, L], [(1-alpha)*L]*2, 'r')


times = np.linspace(0.001, periods*L, 10*periods*L)
rectangles0 = ax0.bar(np.arange(L) + 0.5, np.zeros(L), width=0.8, color='black')  # rectangles to animate
rectanglesl = axl.bar(np.arange(L) + 0.5, np.zeros(L), width=0.9, color='black')  # rectangles to animate
rectanglesr = axr.bar(np.arange(L) + 0.5, np.ones(L)*L/2, width=0.9, color='green')  # rectangles to animate
stats_plot_nback, = axs.plot([],[], label='#back')
stats_plot_nbacknei, = axs.plot([],[], label='#neighbours')
stats_plot_nbackavdist, = axs.plot([],[], label='#av_distance')
axs.legend()
axs.grid()
stats_nback = np.zeros_like(times)
stats_nbacknei = np.zeros_like(times)
stats_nbackavdist = np.zeros_like(times)

# nomen est omen
def init():
    res = animate(0)
    for r0, rl in zip(rectangles0,rectanglesl):
        r0.set_height(rl.get_height())
    return res

# main function
def animate(i):
    t = times[i]
    count_back = 0
    backs = []
    for j, (rl, rr) in enumerate(zip(rectanglesl,rectanglesr)):
        pos, dir = phi(t+shift[j])
        rl.set_height(pos)
        if dir:
            rr.set_color('red')
            count_back += 1
            backs.append(j)
        else:
            rr.set_color('green')
    stats_nback[i] = count_back
    stats_plot_nback.set_data(times[:i], stats_nback[:i])
    backs = np.array(backs)
    diffs = np.diff(backs)
    stats_nbacknei[i] = np.count_nonzero(diffs==1)
    stats_nbackavdist[i] = np.mean(diffs)
    stats_plot_nbacknei.set_data(times[:i], stats_nbacknei[:i])
    stats_plot_nbackavdist.set_data(times[:i], stats_nbackavdist[:i])
    return *rectanglesl, *rectanglesr, stats_plot_nback, stats_plot_nbacknei, stats_plot_nbackavdist

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(times), interval=20, blit=True, repeat=False)

plt.show()
# writergif = animation.PillowWriter(fps=30)
# anim.save(name+".gif", writer=writergif)

