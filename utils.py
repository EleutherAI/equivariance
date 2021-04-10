import numpy as np
from math import floor, sqrt

'''
Defines an affine map

An affine map is a function f(x) = Ax + t for:
A: transformation matrix
t: translation vector
'''
class AffineMap(Object):
    def __init__(self, A, t):
        self.A = A
        self.t = t

    @classmethod
    def fromParams(cls, params):
        '''

        :param params: a list of doubles
        :return: AffineMap
        '''
        dim = cls.getDim(numParams=len(params))
        assert (len(params) == (dim**2 + dim)), "AffineMap dimension should satisfy d^2 + d = numParams"

        A = np.fromfunction(lambda i, j: params[i * dim + j])
        t = np.array([params(dim**2 + i) for i in range(dim)])
        return cls(A, t)

    def getDim(self, numParams):
        '''

        :param numParams: int
        :return: int
        '''
        return int(floor(sqrt(4 * numParams + 1) -1) / 2)

