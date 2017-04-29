import numpy as np


class Spectral(object):
    def __init__(self, dim, kernel, clust, method='Ng'):
        """
        A class for performing spectral clustering of a given dataset.

        **Positional Arguments:**
            - dim:
                - the number of dimensions to use for clustering.
            - kernel:
                - the kernel function to use when computing the affinity matrix.
            - clust:
                - a class for using to cluster the points. The class must contain a
                    'clust.fit(X)' method. This is used to classify the points once they
                    are in the spectral domain. Note that the clust parameter will also
                    define the number of classes we want for our algorithm.
            - method:
                - the method for spectral clustering. Choices are Ng, Bach, and
                    Schiebinger.
        """
        self.kernel = kernel
        self.k = dim
        method_opts = ['Ng', 'Sch', 'Bach']
        if method not in method_opts:
            raise ValueError('You have passed a method that is not supported.\n' +
                             'Supported options are: ' + " ".join(method_opts))
        self.method = method
        self.clust = clust
        self.has_fit = False
        pass

    def fit(self, X):
        """
        A function to fit a spectral clustering model given data.

        **Positional Arguments:**
            - X:
                - the data to fit. Dimensions should be F, n for F features
                    and n data points.
        """
        self.X = X
        if self.method == 'Ng':
            self.__ng_fit__()
        elif self.method == 'Bach':
            self.__bach_fit__()
        elif self.method == 'Sch':
            self.__schiebinger_fit__()
        self.clust.fit(self.Y)
        self.has_fit = True
        pass

    def get_assignments(self):
        if not self.has_fit:
            return ValueError('You have not fit a model yet.\n' +
                              ' try calling fit() first.')
        return 

    def __laplacian__(self):
        """
        A function to implement the normalized Laplacian embedding of a dataset.
        """
        X = self.X
        (self.F, self.n) = X.shape
        A = np.zeros(self.n, self.n)
        for i in range(0, self.n):
            for j in range(0, self.n):
                A[i, j] = self.kernel.dot(X[:,i], X[:,j])
        self.A = A
        D = np.sum(A, axis=1)*np.identity(self.F)
        Dpow = np.linalg.matrix_power(D, -0.5)
        self.L = Dpow.dot(A).dot(Dpow)
        # use the SVD instead of eig here since they are the
        # same for symmetric L, and the SVD will preorder our vectors
        # for us which eig does not
        (U, S, Vs) = np.linalg.svd(self.L)
        return U[:, 0:self.k]

    def __schiebinger_fit__(self):
        """
        A function that implements kernalized spectral clustering
        using the algorithm from Jordan.
        """
        self.X = X
        self.__laplacian__()
        Uk = self.U[:, 0:self.k]
        Unorm = np.linalg.norm(Uk, axis=1)[:, np.newaxis]
        pass

    def __ng_fit__(self):
        """
        A function that implements kernalized spectral clustering
        using the algorithm from Ng.
        """
        self.X = X
        Uk = self.__laplacian__()
        # normalize so that the rows have unit length using matrices
        # for speed
        Unorm = np.linalg.norm(Uk, axis=1)*np.identity(self.n)
        # compute Y and transpose it to get it into kxn
        self.Y = np.linalg.pinv(Unorm).dot(Uk).transpose()
        pass

    def __bach_fit__(self):
        """
        A function that implements kernalized spectral clustering
        using the algorithm from Bach.
        """
        self.X = X
        self.__laplacian__()
        # find the first R eigenvectors
        Ur = self.U[:, 0:self.k]
        pass