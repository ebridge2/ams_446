import numpy as np
from scipy.spatial import distance


class Kmeans(object):
    def __init__(self, K, wt=False, max_iter=100):
        """
        A class that implements K-means clustering. Uses Kmeans++ for initialization.

        **Positional Arguments:**
            - K:
                - the number of clusters.
            - wt:
                - whether to use weighted or unweighted. Defaults to unweighted.
            - max_iter:
                - the number of iterations for defining convergence. This is the
                    maximum number of iterations kmeans will be run.
        """
        self.K = K
        self.max_iter = max_iter
        self.has_fit = False
        self.wt = wt
        self.has_init = False
        pass

    def initialize(self, X, d=None):
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
        self._d = d
        if self.wt:  # if user specified that they wanted weighted, they should
                     # pass in d
            if d is None:
                raise ValueError('user specified weighted kmeans, but did not pass weights.')
        self.__kmeanspp__()
        pass

    def fit(self):
        """
        A function for fitting a k-means model given the initialized data.
        """
        if not self.has_init:
            raise ValueError('The centers have not been initialized yet.\n' +
                             'try using initialize() first.')
        self.__fit__()
        pass

    def __fit__(self):
        """
        A function that uses the standard EM iterative approach
        for k-means.
        """
        converged = False

        # we will need these a bunch, so might as well get them locally
        X = self.X
        max_iter = self.max_iter
        Cent = self.Cent
        n = self.n
        K = self.K

        # for tracking the progress of our algorithm
        niter = 0
        dist = np.zeros(max_iter)
        dist[niter] = np.sum(self.__get_dists__(X, Cent)**2)
        niter += 1
        while not converged and niter < max_iter:
            # expectation step
            # assigned cluster center is the closest center for each point
            assignment = self.__get_assignments__(X, Cent)

            # maximization step
            Cent = self.__maximize_centers__(X, assignment, K)

            # check for convergence as the distance not changing from one iteration
            # to the next
            dist[niter] = self.__compute_distortion__(X, Cent)
            if dist[niter] == dist[niter - 1]:
                converged = True
            niter += 1
        # save the most recent assignments of the points, since this
        # will be a natural thing to want to retrieve.
        self.assignment = assignment
        self.dist = dist
        self.Cent = Cent
        self.niter = niter
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
        n = X.shape[1]
        return np.array([self.__predict__(X[:, i], Cent) for i in range(0, n)])

    def __maximize_centers__(self, X, assignment, K):
        """
        A function to maximize the centers on a given iteration of k-means.
        """
        if not self.wt:
            Cent = [X[:, assignment == c].mean(axis=1) for c in range(0, K)]
        else:
            Cent = []
            for c in range(0, K):
                d_assigned = self._d[assignment == c]
                num = X[:, assignment == c].dot(np.reciprocal(np.sqrt(d_assigned))*np.identity(len(d_assigned)))
                Cent.append(num.sum(axis=1)/d_assigned.sum())
        return Cent

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
        """
        A function to return the distance between a point and all of the centers.
        """
        return np.array([distance.euclidean(x, c) for c in Cent])

    def __get_dists__(self, X, Cent):
        """
        A function to return the distance to the closest center for each point.
        """
        n = X.shape[1]
        if self.wt:  # if weighted, weight the points before finding centers
            X = X.dot(np.reciprocal(np.sqrt(self._d))*np.identity(self.n))
        return np.array([self.__dist__(X[:, i], Cent).min() for i in range(0, n)])

    def __compute_distortion__(self, X, Cent):
        """
        A function to compute the distortion metric, a potential convergence
        criterion for Kmeans.
        """
        if not self.wt:
            return np.sum(self.__get_dists__(X, Cent)**2)
        else:
            return self._d.dot(self.__get_dists__(X, Cent)**2)

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
        A function to return the centers at least after initialization has taken place.
        """
        if not self.has_init:
            raise ValueError('The centers have not been initialized yet.\n' +
                             'try using initialize() first.')
        return np.vstack(self.Cent).transpose()

    def __kmeanspp__(self):
        """
        A function for improving the initial centers for the k-means algorithm
        using the strategy found: https://en.wikipedia.org/wiki/K-means%2B%2B.
        This strategy is called K-means++ initialization.
        """
        X = self.X
        # our centers
        Cent = [X[:, np.random.randint(low=0, high=self.n)]]
        for k in range(1, self.K):
            # array for the squared distances to closest nearby cluster per point
            D = self.__get_dists__(X, Cent)
            Ds = np.square(D)
            # choose a point at random given the probability distribution Dsquared
            # and define as our new center
            Ds = Ds/float(np.sum(Ds))
            Cent.append(X[:, np.random.choice(range(0, self.n), p=Ds)])
        self.Cent = Cent
        self.has_init = True
        pass
