import numpy as np
from scipy.spatial import distance
from abc import ABCMeta


class Kernel(object):
    """
    A class containing kernel functions to be used in conjunction with clustering package.
    Defaults to a linear kernel.
    """
    __metaclass = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dot(xi, xj):
        return xi.dot(xj)


def RBF_kernel(Kernel):
    """
    A class containing an RBF kernel.
    """
    def __init(self, sigma):
        super(Kernel, self).__init__()
        self.sigma = sigma
        pass

    def dot(xi, xj):
        exp = np.linalg.norm(xi - xj)**2/float(2*sigma**2)
        return np.exp(exp)

