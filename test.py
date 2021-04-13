import numpy as np
from maps import AffineMap

'''
Affine Map Tests
'''
def test_apply():
    '''
    Tests applying the affine map to a point
    :return:
    '''
    A = np.array([[1,0],[0,1]])
    t = np.array([5,5])
    x = np.array([1,2])
    map = AffineMap(A, t)
    output = map.apply(x)
    assert output[0] == 6 and output[1] == 7, "Pure translation map must apply correctly"


