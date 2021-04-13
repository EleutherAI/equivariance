import numpy as np
from math import floor, sqrt
from dataclasses import dataclass

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
        assert self.A.shape[0] == self.t.shape[0], "Dimensions of matrices in AffineMap must match"

    def apply(self, x):
        '''
        Applies the affine map to an input. Returns y = Ax + t
        :param x: numpy array
        :return: numpy array
        '''
        return self.A @ x + self.t

class Similitude(AffineMap):
    def __init__(self):
        pass