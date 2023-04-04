import numpy as np
from functools import reduce
from scipy.signal import savgol_filter


#------------------------------------------------------------------
def add_neighbors(refine, n):
    refine = reduce(np.union1d, (refine, refine - 1, refine + 1))
    return np.setdiff1d(refine, [-1, n-1], assume_unique=True)


#------------------------------------------------------------------
def adapt_mesh(t, eta, theta=0.6, plot=False, filter=False, thetafac=1.1, refneighbors=False, doubleref=False):
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=plt.figaspect(1))
        plt.plot(eta, label="orig")
    if filter:
        eta = savgol_filter(eta, min(5,len(eta)), 1)
    if plot:
        plt.plot(eta, label="savgol")
        plt.legend()
        plt.grid()
        plt.show()
    indsort = np.argsort(eta)[::-1]
    estacc = np.add.accumulate(eta[indsort])
    etatotal = estacc[-1]
    index = np.searchsorted(estacc, theta*etatotal)
    estaccval = estacc[index]
    # print(f"{estacc=} {index=} {estaccval=} {theta*estacc[0]=}")
    if thetafac!=1: index = np.searchsorted(estacc, min(thetafac*estaccval,theta*etatotal))
    nt = len(t)
    if doubleref:
        meanvalue = 0.25*(estaccval + estacc[0])
        # print(f"{estaccval=} {estacc[0]=} {meanvalue=}")
        indexmean = np.searchsorted(estacc, meanvalue)
        # print(f"{index=} {indexmean=}")
        assert indexmean <= index
        refine_double = indsort[:indexmean]
        refine = indsort[indexmean:index+1]
        tnew = np.empty(nt+len(refine)+3*len(refine_double))
        tnew[:nt] = t
        nr = len(refine)
        nrd = len(refine_double)
        # print(f"{nr=} {nrd=}")
        tnew[nt + np.arange(nr)] = 0.5 *(t[refine]+t[refine+1])
        dt = t[refine_double+1]-t[refine_double]
        tnew[nt + nr + 3*np.arange(nrd)] = t[refine_double] + 0.25 *dt
        tnew[nt + nr + 3*np.arange(nrd)+1] = t[refine_double] + 0.5 *dt
        tnew[nt + nr + 3*np.arange(nrd)+2] = t[refine_double] + 0.75 *dt
    else:
        refine = indsort[:index+1]
        if refneighbors: refine = add_neighbors(refine, nt)
        tnew = np.empty(nt+len(refine))
        tnew[:nt] = t[:]
        tnew[nt + np.arange(len(refine))] = 0.5 *(t[refine]+t[refine+1])
    tnew.sort()
    return tnew, {"nref":index+1, "nperc":(len(tnew)-nt)/nt, "theta_used":estacc[index]/etatotal}


