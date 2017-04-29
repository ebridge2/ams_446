import numpy as np
from scipy.spatial import distance
from abc import ABCMeta, abstractmethod


class Kernel(object):
    """
    A class that implements a kernel by computing an inner product
    between two vectors using a similarity function. This similarity
    function can map the vectors to nonstandard feature spaces.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dot(xi, xj):
        pass


class Linear_kernel(Kernel):
    """
    A class implementing a linear kernel.
    """
    def __init(self):
        super(Kernel, self).__init__()
        pass

    def dot(xi, xj):
        return xi.dot(xj)


class RBF_kernel(Kernel):
    """
    A class implementing an RBF kernel.
    """
    def __init(self, sigma):
        super(Kernel, self).__init__()
        self.sigma = sigma
        pass

    def dot(xi, xj):
        exp = np.linalg.norm(xi - xj)**2/float(2*sigma**2)
        return np.exp(exp)

