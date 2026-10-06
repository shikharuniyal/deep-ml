import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	
	X = np.array(points,dtype = float)
	mu = np.array(initial_centroids,dtype = float)
	n = X.shape[0]
	d = X.shape[1]
	oned = np.full((d,1),1,dtype = float)

	for i in range(max_iterations):
		#by hardman product
		sx = (X*X) @ oned
		sm = (mu*mu) @ oned
		D = sx  - 2* X.dot(mu.T)  + sm.T
		c = D.argmin(axis  =1)# crazyyy just like math
		#c is list
		R = np.full((n,k),0)
		for i in range(n):	
			R[i][c[i]] = 1
		#----update fix r optimize mu
		sums = R.T.dot(X)
		sizes  = R.sum(axis = 0)

		new_mu = mu.copy()                    # keeps old centroid if a cluster is empty
        for j in range(k):
            if sizes[j] > 0:
                new_mu[j] = sums[j] / sizes[j]

		mu = new_mu

	res = [tuple(row) for row in mu]
	return res