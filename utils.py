import numpy as np
from math import sin, cos

def construct_rotation_2d(angle):
    '''
    Constructs an appropriate rotation matrix from a list of angles in radians
    :param angles: List of angles
    :return:
    '''
    s = sin(angle)
    c = cos(angle)
    return np.array([[c, -s],[s, c]])

