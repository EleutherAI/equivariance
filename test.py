import numpy as np
from utils import AffineMap

'''
Affine Map Tests
'''
def testFromParams():
    # test case taken by passing in
    l = [0.5, 0.0, 0.0, 0.5, 0.0, -0.5]
    AffineMap.fromParams(l)

testFromParams()
