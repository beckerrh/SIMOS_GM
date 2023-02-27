class AlgoInput():
    def __init__(self, tol=1e-6, maxiter=1000, history=False, nbt=20, omegabt=0.7, alphabt=0.01):
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = True
        self.history = history
        self.nbt = nbt
        self.omegabt = omegabt
        self.alphabt = alphabt