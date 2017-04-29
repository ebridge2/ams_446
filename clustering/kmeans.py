import numpy as np
from scipy.spatial import distance


class kmeans(object):
    def __init__(self, K, max_iter=100):
        """
        A class that implements K-means clustering. Uses Kmeans++ for initialization.

        **Positional Arguments:**
            - niter:
                - the number of iterations for defining convergence. This is the
                    maximum number of iterations kmeans will be run.
            - K:
                - the number of clusters.
        """
        self.tol = tol
        self.K = K
        self.max_iter = max_iter
        self.has_fit = False
        self.has_init = False
        pass

    def initialize(self, X):
        """
        A function to initialize our kmeans instance, using kmeans++.

        **Positional Arguments:**
            - X:
                - the data to fit. Dimensions should be F, n for F features
                    and n data points.
        """
        self.X = X
        (self.F, self.n) = X.shape
        # initialize w Kmeans++
        self.__kmeanspp__()
        pass

    def fit(self):
        """
        A function that uses the standard EM iterative approach
        for k-means.
        """
        if not self.has_init:
            raise ValueError('The centers have not been initialized yet.\n' +
                             'try using initialize() first.')
        converged = False

        # we will need these a bunch, so might as well get them locally
        max_iter = self.max_iter
        Cent = self.Cent
        n = self.n
        K = self.K

        # for tracking the progress of our algorithm
        niter = 0
        dist = np.zeros(max_iter)
        while not converged and niter < max_iter:
            # expectation step
            # assigned cluster center is the closest center for each point
            assignment = self.__get_assignments__(X, Cent)

            # maximization step
            Cent = [X[:, assignment == c].mean(axis=1) for c in range(0, len(K))]

            # check for convergence as the distance not changing from one iteration
            # to the next
            dist[niter] = np.sum(np.array([self.__dist__(X[:, i], Cent).min() for i in range(0, n)]))
            if dist[niter] == dist[niter - 1]:
                converged = True
            niter += 1
        # save the most recent assignments of the points, since this
        # will be a natural thing to want to retrieve.
        self.assignment = assignment
        self.dist = dist
        self.Cent = Cent
        # mark that we have fit our centers so that our algorithm will now
        # be able to make predictions
        self.has_fit = True
        pass

    def __get_assignments__(self, X, Cent):
        """
        A function to return the assignments of a matrix of data.
        We define a private version so that users can't get assignments for
        arbitrary centers.
        """
        return np.array([self.__predict__(X[:, i], Cent) for i in range(0, n)])

    def get_assignments(self, X):
        """
        Returns the assignments for a large amount of data at once in matrix form.

        **Positional Arguments:**
            - X:
                - the data to fit. Dimensions should be F, n for F features
                    and n data points.        
        """
        if not self.has_init:
            raise ValueError('The centers have not been initialized yet.\n' +
                             'try using initialize() first.')
        return self.__get_assignments__(X, self.Cent)

    def __dist__(self, x, Cent):
        return np.array([distance.euclidian(x, c) for c in Cent])

    def __predict__(self, x, Cent):
        """
        Again, same deal: we don't want users to be able to specify
        arbitrary centers.
        """
        return self.__dist__(x, Cent).argmin()

    def predict(self, x):
        """
        A function for making predictions given a trained K-means instance.

        **Positional Arguments:**
            - x:
                a new data point to assign a cluster to.
        """
        if not self.has_fit:
            raise ValueError('The model is not yet trained.\n' +
                              'try using fit() first.')
        return self.__predict__(x, self.Cent)

    def get_centers(self):
        """
        A function to return the centers after initialization has taken place.
        """
        if not self.has_init:
            raise ValueError('The centers have not been initialized yet.\n' +
                             'try using initialize() first.')
        return self.Cent

    def __kmeanspp__(self):
        """
        A function for improving the initial centers for the k-means algorithm
        using the strategy found: https://en.wikipedia.org/wiki/K-means%2B%2B.
        This strategy is called K-means++ initialization.
        """
        # our centers
        Cent = [X[:, np.random.randint(low=0, high=self.n)]]
        for k in range(0, self.K):
            # array for the squared distances to closest nearby cluster per point
            Ds = np.zeros(self.n)
            for i in range(0, self.n):
                # pick D[i] as the distance to the closest center already chosen
                Ds[i] = np.array([distance.euclidian(X[:, i], c) for c in Cent]).min()**2
            # choose a point at random given the probability distribution Dsquared
            # and define as our new center
            Cent.append(X[:, np.random.choice(range(0, self.n), p=Ds))]
        self.Cent = Cent
        self.has_init = True
        pass