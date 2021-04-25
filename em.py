import numpy as np
from maps import *
from itertools import product

class EM():
    def __init__(self, maps=None, weights=None, depth=100):
        self.maps = maps
        self.weights = weights
        self.depth_weights = np.log(np.ones((depth,)))
        self.depth = depth
        self.post_transform = None

    def train(self, data):
        if not self.maps:
            self.maps, self.weights = self.create_initial_model()

        num_mixtures = len(self.maps)
        dim = data.shape[1]

        pass

    def iter_once(self, data):
        pass

    def create_initial_model(self):
        pass

    def e_step(self, data):
        '''
        This computes the expectation step
        The data is modeled as a mixture of gaussians, weighted by the depth weights and the code probabilities
        The gaussians themselves are constructed by composing the post-transform Similitude with the IFS components
        indicated by the code and using that function to transform a standard normal
        :param data:
        :return:
        '''
        # apply the inverse post transform so we can just deal with the code IFS components
        inverse_map = self.post_transform.invert()
        transformed_data = np.apply_along_axis(inverse_map.apply, 1, data) # transforms each data pt by the inverse map todo test



        base_scale = 1
        base_similitude = AffineMap(np.eye(data.shape[1]), np.zeros((data.shape[1]),))
        pass

    def m_step(self, data):
        pass

    def codes_at_depth(self, vals, depth):
        return [list(x) for x in product(vals, repeat=depth)]

    def compute_code_values(self, dim):
        '''

        :return:
        '''
        codons = np.arange(len(self.weights))
        codes = []
        depth_probs = []
        code_weights = []
        scalars = []
        translations = []
        for i in range(self.depth):
            codes += self.codes_at_depth(codons, i)

        for code in codes:
            code_log_prob = np.log(self.weights[code]).sum()
            depth_log = self.depth_weights[len(code)]
            code_weights.append(code_log_prob)
            depth_probs.append(depth_log)

            scalar = 1
            base_trans = np.zeros((dim,))
            for m in self.maps[code]:
                scalar *= m.scalar
                base_trans = m.apply(base_trans) # todo test this

            translations.append(base_trans)
            scalars.append(scalar)
        
        return np.array(codes), np.array(code_weights), np.array(depth_probs), np.array(scalars), np.array(translations)



