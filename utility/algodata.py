class AlgoInput():
    def __init__(self, tol=1e-6, maxiter=1000):
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = True
        self.history = False
        self.nbt = 20
        self.omegabt = 0.8
        self.alphabt = 0.1