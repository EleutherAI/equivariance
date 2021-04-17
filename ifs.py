import numpy as np
from maps import *
from scipy.stats import multivariate_normal
def generate_ifs(maps, weights, depth, size):
    '''

    :param maps:
    :param weights:
    :param depth:
    :return:
    '''
    assert sum(weights) == 1, "Weights for IFS must sum to 1"
    dim = maps[0].transform.shape[0]
    norm = multivariate_normal(mean=np.zeros((dim, )), cov=np.eye(dim))
    out_points = np.array([ifs_iter_once(maps, weights, norm, depth) for i in range(size)])
    return out_points

def ifs_iter_once(maps, weights, base_dist, depth):
    '''
    Applies the IFS at the specified depth for one output datapoint.
    :param maps:
    :param weights:
    :return:
    '''
    sample_p0 = base_dist.rvs() # draw a random starting point
    for i in range(depth):
        # in each iter, draw a random map with weights as probabilities and apply it to the sample point
        map = np.random.choice(a=maps, p=weights)
        sample_p0 = map.apply(sample_p0)

    return sample_p0
