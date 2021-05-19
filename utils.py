import numpy as np
from math import sin, cos, pi
from maps import Similitude

def construct_rotation_2d(angle):
    '''
    Constructs an appropriate rotation matrix from a list of angles in radians
    :param angles: List of angles
    :return:
    '''
    s = sin(angle)
    c = cos(angle)
    return np.array([[c, -s],[s, c]])

def identity_similitude(dim):
    '''

    :param dim:
    :return:
    '''
    trans = np.zeros((dim, ))
    rot = np.eye(dim)
    return Similitude(1.0, rot, trans)

def sphere_init(dim, comps, radius, scale):
    '''

    :param dim:
    :param comps:
    :param radius:
    :param scale:
    :return:
    '''
    # sample points in the unit sphere
    pts = []
    for i in range(comps):
        rand = np.random.normal(0,1,(dim,))
        norm = np.linalg.norm(rand)
        rand = rand / norm
        rand = rand * radius
        pts.append(rand)

    sims = []
    for point in pts:
        trans = point * (1 - scale)
        angle = np.random.sample() * pi * 2
        rot = construct_rotation_2d(angle)
        sim = Similitude(scale, rot, trans)
        sims.append(sim)

    return sims