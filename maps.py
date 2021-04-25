import numpy as np
from math import floor, sqrt
from dataclasses import dataclass

'''
Defines an affine map

An affine map is a function f(x) = Ax + t for:
transform: transformation matrix
t: translation vector
'''
class AffineMap():
    def __init__(self, A, t):
        self.transform = A
        self.t = t
        assert self.transform.shape[0] == self.t.shape[0], "Dimensions of matrices in AffineMap must match"

    def apply(self, x):
        '''
        Applies the affine map to an input. Returns y = Ax + t
        :param x: numpy array
        :return: numpy array
        '''
        return self.transform @ x + self.t

    def invert(self):

        inverse = np.linalg.inv(self.transform)
        translation = -1 * (inverse @ self.t)
        AffineMap(inverse, translation)

class Similitude(AffineMap):
    def __init__(self, scalar, rotation, translation):
        self.scalar = scalar
        self.transform = scalar * rotation
        self.t = translation

