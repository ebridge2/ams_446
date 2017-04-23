import numpy as np


class spectral(object):
    def __init__(self, kernel, dim):
        self.kernel = kernel
        self.k = dim

    def laplacian(self):
        """
        A function to implement the normalized Laplacian embedding of a dataset.
        """
        X = self.X
        self.f = X.shape[0]
        self.n = X.shape[1]
        A = np.zeros(self.n, self.n)
        for i in range(0, self.n):
            for j in range(0, self.n):
                A[i, j] = 1/float(self.n)*kernel(X[:,i], X[:,j])
        self.A = A
        D = np.sum(D, axis=1)
        Dpow = np.linalg.matrix_power(D, -0.5)
        self.L = Dpow.dot(A).dot(Dpow)
        (self.U, self.S) = np.linalg.eig(self.L)
        pass

    def schiebinger_fit(self, X):
        """
        A function that implements kernalized spectral clustering using the algorithm from Jordan.
        """
        self.X = X
        self.laplacian()
        Uk = self.U[:, 0:self.k]
        Unorm = np.linalg.norm(Uk, axis=1)[:, np.newaxis]
        pass

    def ng_fit(self, X):
        """
        A function that implements kernalized spectral clustering using the algorithm from Ng.
        """
        self.X = X
        self.laplacian()
        # find the k largest eigenvectors of L
        Uk = self.U[:, 0:self.k]
        # form the matrix Y from X by renormalizing each of Xs rows to have unit length
        Unorm = np.linalg.norm(Uk, axis=1)[:, np.newaxis]
        Y = Uk / Unorm
        pass

    def bach_fit(self, X):
        """
        A function that implements kernalized spectral clustering using the algorithm from Bach.
        """
        self.X = X
        self.laplacian()
        # find the first R eigenvectors
        Ur = self.U[:, 0:self.k]
        pass
