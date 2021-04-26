import numpy as np
from maps import *
from itertools import product
from functools import cached_property

class EM():
    def __init__(self, maps=None, weights=None, depth=100):
        self.maps = maps
        self.weights = weights
        self.depth_weights = np.log(np.ones((depth,)))
        self.depth = depth
        self.post_transform = None
        self.pk_map = {}

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
        h = data.shape[1]
        transformed_data = np.apply_along_axis(inverse_map.apply, 1, data) # transforms each data pt by the inverse map todo test
        code_weights, depth_probs, scalars, translations = self.compute_code_values(data.shape[0], data.shape[1])
        transformed_data = np.tile(transformed_data, (1,1,len(self.codes)))

        X_diff = transformed_data - translations
        scale_factor = -1 * h * np.log(scalars)

        # here we want to take the dot product of each row with itself and sum along the middle axis to end with dim nxm
        dot_product = np.einsum('ijk,ijk-> ik', X_diff, X_diff)

        # -hlog(s) - (X dot X) / (2 * s^2)
        norm_log_prob = scale_factor - (np.divide(dot_product, 2 * np.square(scalars)))

        p = depth_probs + code_weights + norm_log_prob

        # normalize rows
        sums = p.sum(axis = 1)
        p = p / sums[:, np.newaxis]

        # pull out the Pk submatrices (columns associated with codes that start with k for each k)
        pks = []
        for i in range(len(self.weights)):
            pks.append(p[:, self.pk_map[i]])

        return p, pks

    def m_step(self, data):
        pass

    def update_depths(self, data, p):
        '''
        Sum the elements in P for the columns j where len(codes[j]) == d for each depth
        :param data:
        :param p:
        :return:
        '''
        d_inds = {}
        for j in range(len(self.codes)):
            c = self.codes[j]
            d = len(c)
            if d in d_inds:
                d_inds[d].append(j)
            else:
                d_inds[d] = [j]

        for l in range(self.depth):
            Pij = p[:, d_inds[l]]
            logsum = np.log(np.sum(Pij))
            self.depth_weights[l] = logsum

    def update_t(self):
        pass



    def codes_at_depth(self, vals, depth):
        return [list(x) for x in product(vals, repeat=depth)]

    @cached_property
    def codes(self):
        '''
        The codes only depend on the number of maps, which should be constant, so we compute once and cache
        :return:
        '''
        codons = np.arange(len(self.weights))
        codes = []
        pk_map = {}

        # precompute the map for the indices of codes that start with k for each k in 0 to the number of maps
        for i in range(self.depth):
            codes_temp = self.codes_at_depth(codons, i)
            codes += codes_temp
            for j in range(len(codes_temp)):
                code = codes_temp[j]
                if code[0] in pk_map:
                    pk_map[code[0]].append(len(codes) + j)
                else:
                    pk_map[code[0]] = [len(codes) + j]

        self.pk_map = pk_map
        self.codes = np.array(codes)

    def compute_code_values(self, data_dim, dim):
        '''

        :return:
        '''
        depth_probs = []
        code_weights = []
        scalars = []
        translations = []

        for code in self.codes:
            code_log_prob = np.log(self.weights[code]).sum()
            depth_log = self.depth_weights[len(code)]
            code_weights.append(code_log_prob)
            depth_probs.append(depth_log)

            scalar = 1
            base_trans = np.zeros((dim,))
            for m in self.maps[code]:
                scalar *= m.scalar
                base_trans = m.apply(base_trans) # todo test this

            tiled = np.tile(base_trans, (data_dim, 1))
            translations.append(tiled)
            scalars.append(scalar)

        code_weights = np.array(code_weights)
        depth_probs = np.array(depth_probs)
        scalars = np.array(scalars)
        translations = np.stack(translations, axis = 2)

        # get dimensions right, want there to be (n, m) where m = len(codes)
        # codes = np.tile(codes, (data_dim,1))
        code_weights = np.tile(code_weights, (data_dim,1))
        depth_probs = np.tile(depth_probs, (data_dim,1))
        scalars = np.tile(scalars, (data_dim,1))

        return code_weights, depth_probs, scalars, translations



