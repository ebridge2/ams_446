import numpy as np


class Spectral(object):
    def __init__(self, dim, kernel, clust, method='Ng'):
        """
        A class for performing spectral _clustering of a given dataset.

        **Positional Arguments:**
            - dim:
                - the number of dimensions to use for _clustering.
            - kernel:
                - the kernel function to use when computing the affinity matrix.
            - _clust:
                - a class for using to _cluster the points. The class must contain a
                    '_clust.fit(X)' method. This is used to classify the points once they
                    are in the spectral domain. Note that the _clust parameter will also
                    define the number of classes we want for our algorithm.
            - method:
                - the method for spectral _clustering. Choices are Ng, Bach, and
                    Schiebinger.
        """
        self.kernel = kernel
        self.k = dim
        method_opts = ['Ng', 'Sch', 'Bach']
        if method not in method_opts:
            raise ValueError('_You have passed a method that is not supported.\n' +
                             'Supported options are: ' + " ".join(method_opts))
        self.method = method
        self._clust = clust
        self.has_fit = False
        pass

    def fit(self, X):
        """
        A function to fit a spectral _clustering model given data.

        **Positional Arguments:**
            - X:
                - the data to fit. Dimensions should be F, n for F features
                    and n data points.
        """
        self._X = X
        if self.method == 'Ng':
            self.__ng_fit__()
        elif self.method == 'Bach':
            self.__bach_fit__()
        elif self.method == 'Sch':
            raise ValueError('not implemented yet!')
            self.__schiebinger_fit__()
        self._clust.initialize(self._Y)
        self._clust.fit()
        self.has_fit = True
        pass

    def get_assignments(self):
        """
        A function to return the assignments of the training data.
        """
        if not self.has_fit:
            raise ValueError('_You have not fit your model yet.\n' +
                             'try calling fit() first.')
        return self._clust.get_assignments(self._Y)

    def spectral_data(self):
        """
        A function to return the training data in the spectral domain.
        """
        if not self.has_fit:
            raise ValueError('_You have not fit your model yet.\n' +
                             'try calling fit() first.')
        return self._Y

    def spectral_centers(self):
        """
        A function to return the centers in the spectral domain.
        """
        if not self.has_fit:
            raise ValueError('_You have not fit your model yet.\n' +
                             'try calling fit() first.')
        return self._clust.get_centers()

    def __laplacian__(self):
        """
        A function to implement the normalized Laplacian embedding of a dataset.
        """
        X = self._X
        (self._F, self._n) = X.shape
        A = np.zeros((self._n, self._n))
        for i in range(0, self._n):
            for j in range(0, self._n):
                A[i, j] = self.kernel.dot(X[:,i], X[:,j])
        self.A = A
        D = np.sum(A, axis=1)*np.identity(self._n)
        # D^(-.5) is the element wise reciprocal of the sqrt
        Dpow = np.linalg.pinv(np.sqrt(D))
        self._L = Dpow.dot(A).dot(Dpow)
        # use the SVD instead of eig here since they are the
        # same for symmetric L, and the SVD will preorder our vectors
        # for us which eig does not
        (U, S, Vs) = np.linalg.svd(self._L)
        return U[:, 0:self.k]

    def get_laplacian(self):
        """
        A function to return the laplacian matrix of the data.
        """
        if not self.has_fit:
            raise ValueError('_You have not fit your model yet.\n' +
                             'try calling fit() first.')
        return self._L          

    def __schiebinger_fit__(self):
        """
        A function that implements kernalized spectral _clustering
        using the algorithm from Jordan.
        """
        self.__laplacian__()
        Uk = self.U[:, 0:self.k]
        Unorm = np.linalg.norm(Uk, axis=1)[:, np.newaxis]
        pass

    def __ng_fit__(self):
        """
        A function that implements kernalized spectral _clustering
        using the algorithm from Ng.
        """
        Uk = self.__laplacian__()
        # normalize so that the rows have unit length using matrices
        # for speed
        Unorm = np.linalg.norm(Uk, axis=1)*np.identity(self._n)
        # compute _Y and transpose it to get it into kxn
        self._Y = Uk.transpose().dot(np.linalg.pinv(Unorm))
        pass

    def __bach_fit__(self):
        """
        A function that implements kernalized spectral _clustering
        using the algorithm from Bach.
        """
        Uk = self.__laplacian__()
        # find the first R eigenvectors
        self._Y = Uk.transpose()
        pass
