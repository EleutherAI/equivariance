import numpy as np
from maps import *

def generate_ifs(maps, weights, depth, size):
    '''

    :param maps:
    :param weights:
    :param depth:
    :return:
    '''
    assert sum(weights) == 1, "Weights for IFS must sum to 1"
    basis = depth
    norm = maps[0].transform.shape[0]
    pass

def ifs_iter_once(maps, weights, base_dist, depth):
    '''

    :param maps:
    :param weights:
    :return:
    '''
    sample_p0 = base_dist.rvs()
    for i in range(depth):
        map = np.random.choice(a=maps, p=weights)
        sample_p0 = map.apply(sample_p0)

    return sample_p0
