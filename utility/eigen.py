import numpy as np

n=4
a = np.ones(n-1)
A = np.zeros(shape=(n,n))
i = np.arange(n-1)
mus = np.array([2/(2*i+1) for i in range(n)])/2
A[0,0] = 1
A[-1,-1] = 1-2/np.sqrt(3)
A[i+1,i] = mus[1:]
A[i,i+1] = -mus[:-1]
np.set_printoptions(precision=3)
print(f"A=\n{A}\n\n {np.linalg.eigvals(A)}")
