import numpy as np
from math import floor, sqrt

'''
Defines an affine map

An affine map is a function f(x) = Ax + t for:
A: transformation matrix
t: translation vector
'''


def getDim(numParams):
    '''

    :param numParams: int
    :return: int
    '''
    return int(floor(sqrt(4 * numParams + 1) - 1) / 2)

class AffineMap():
    def __init__(self, A, t):
        self.A = A
        self.t = t

    @classmethod
    def fromParams(cls, params):
        '''

        :param params: a list of doubles
        :return: AffineMap
        '''
        dim = getDim(len(params))
        assert (len(params) == (dim**2 + dim)), "AffineMap dimension should satisfy d^2 + d = numParams"

        A = np.reshape(np.array(params[: dim**2]), (dim, dim))
        t = np.array([params[dim**2 + i] for i in range(dim)])
        return cls(A, t)


