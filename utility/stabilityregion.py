import numpy as np
import matplotlib.pyplot as plt

def p(z, theta, k):
    if k==1:
        return (1-(1-theta)*z/2)/(1+(1+theta)*z/2)
    elif k==2:
        return (1 - z/2 + (1 - theta)*z**2/12) / (1 + z/2 + (1 + theta)*z**2/12)
    elif k == 3:
        return (1 - z / 2 + z ** 2 / 10 - (1 - theta) * z ** 3 / 120) / (
                    1 + z / 2 + z ** 2 / 10 + (1 + theta) * z ** 3 / 120)
    elif k==4:
        return (1 - z/2 + z**2*(3/28) - z**3/84+ (1 - theta) *z**4/1680) / (1 + z/2 + z**2*(3/28) + z**3/84 + (1 + theta) *z**4/1680)
    raise ValueError(f"{k=} not implemented")
#plt.imshow(np.abs(p(z))<=1)


def plot_stabregion(thetas, ks, a=20, n=300):
    lsp = np.linspace(-a, a, n)
    z = lsp[:, np.newaxis] + 1j * lsp
    fig,axs=plt.subplots(len(thetas),len(ks), figsize=plt.figaspect(len(thetas)/len(ks)), sharex=True, sharey=True)
    for i,theta in enumerate(thetas):
        for k in ks:
            ax = axs[i,k-1]
            ax.contourf(np.real(z), np.imag(z), np.abs(p(z,theta=theta, k=k))<=1, levels=1)
            if k==1: ax.set_ylabel(r'$\theta$='+f"{theta}")
            if i==len(thetas)-1: ax.set_xlabel(f"{k=}")
    plt.show()

def plot_dissipation(thetas, ks, a=20, n=300):
    z = np.linspace(0, a, n)
    fig,axs=plt.subplots(len(thetas),len(ks), figsize=plt.figaspect(len(thetas)/len(ks)), sharex=True, sharey=True)
    for i,theta in enumerate(thetas):
        for k in ks:
            ax = axs[i,k-1]
            ax.plot(z, p(z,theta=theta, k=k))
            ax.plot(z, np.exp(-z), '--r')
            if k==1: ax.set_ylabel(r'$\theta$='+f"{theta}")
            if i==len(thetas)-1: ax.set_xlabel(f"{k=}")
    plt.show()

thetas = [0, 0.5, 1]
ks = [1,2,3,4]
# plot_stabregion(thetas, ks)
plot_dissipation(thetas, ks)