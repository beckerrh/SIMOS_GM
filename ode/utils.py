import numpy as np
from functools import reduce
from scipy.signal import savgol_filter


def add_neighbors(refine, n):
    refine = reduce(np.union1d, (refine, refine - 1, refine + 1))
    return np.setdiff1d(refine, [-1, n-1], assume_unique=True)


#------------------------------------------------------------------
def adapt_mesh(t, eta, theta=0.6, plot=False, filter=False, thetafac=1.1, refneighbors=False, doubleref=True):
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


#------------------------------------------------------------------
def adapt_mesh_old(t, eta, theta=0.6, plot=False, filter=False, thetafac=1.1, refneighbors=False, doubleref=False):
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
    indsort = np.argsort(eta)
    # print(f"{list(esttot)=}\n{indsort=}")
    estacc = np.add.accumulate(eta[indsort[::-1]])
    # index = np.argmax(estacc > theta*estacc[-1])
    index1 = np.searchsorted(estacc, theta*estacc[-1])
    estaccval = estacc[index1]
    if estaccval < theta*estacc[-1]:
        raise ValueError(f"strange: {index1=} {theta*estacc[-1]=} {estaccval=}\n {estacc=}")
    # print(f"{index1=} {theta*estacc[-1]=} {estaccval=}")
    if thetafac!=1: index = np.searchsorted(estacc, min(thetafac*estaccval,estacc[-1]))
    else: index=index1
    # print(f"nref={index} nperc={index/len(esttot):5.2f} {theta=} theta_used={estacc[index]/estacc[-1]:5.2f}")
    nt = len(t)
    # if doubleref:
    #     meanvalue = 0.5*(estacc[index] + estacc[-1])
    #     # print(f"{estacc[index]=} {estacc[-1]=}")
    #     indexmean = np.searchsorted(estacc, meanvalue)
    #     refine_double = indsort[indexmean:]
    #     refine = indsort[index:indexmean]
    tnew = np.empty(nt+index+1)
    tnew[:nt] = t
    for i in range(index+1):
        ii = indsort[-1-i]
        tnew[nt+i] = 0.5*(t[ii]+t[ii+1])
    refine = indsort[-index-1:]
    tnew2 = np.empty(nt+len(refine))
    for i in range(len(refine)):
        ii = refine[i]
        tnew2[nt + i] = 0.5 * (t[ii] + t[ii + 1])
    assert np.allclose(tnew, tnew2)

    # if doubleref:
    #     meanvalue = 0.5*(estacc[index] + estacc[-1])
    #     # print(f"{estacc[index]=} {estacc[-1]=}")
    #     indexmean = np.searchsorted(estacc, meanvalue)
    #     refine_double = indsort[indexmean:]
    #     refine = indsort[index:indexmean]
    #     # refine_double = indsort[index:]
    #     # refine = indsort[-1:-1]
    #     # refine = indsort[index:]
    #     # refine_double = indsort[-1:-1]
    #
    #     # print(f"{index=} {indexmean=}\n{refine=}\n{refine_double=}")
    #     tnew = np.empty(nt+len(refine)+3*len(refine_double))
    #     tnew[:nt] = t
    #     nr = len(refine)
    #     for i in range(nr):
    #         ii = refine[i]
    #         tnew[nt + i] = 0.5 * (t[ii] + t[ii + 1])
    #     for i in range(len(refine_double)):
    #         ii = refine_double[i]
    #         dt = t[ii + 1] - t[ii]
    #         tnew[nt + nr + 3*i] = t[ii] + 0.25*dt
    #         tnew[nt + nr + 3*i+1] = t[ii] + 0.5*dt
    #         tnew[nt + nr + 3*i+2] = t[ii] + 0.75*dt
    # else:
    #     # refine = indsort[-1-index:]
    #     refine = indsort[index:]
    #     if refneighbors:
    #         refine = add_neighbors(refine, nt-1)
    #     tnew = np.empty(nt+len(refine))
    #     tnew[:nt] = t
    #     for i in range(len(refine)):
    #         ii = refine[i]
    #         tnew[nt + i] = 0.5 * (t[ii] + t[ii + 1])
    tnew.sort()
    # catastrophe:
    # tnew[1:-1] = 0.5*(tnew[2:]+tnew[:-2])
    # print(f"{list(t)=}\n{list(tnew)=}")
    return tnew, {"nref":index+1, "nperc":len(tnew)/len(eta), "theta_used":estacc[index]/estacc[-1]}
