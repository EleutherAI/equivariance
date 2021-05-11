import numpy as np
from maps import *
from itertools import product
from cached_property import cached_property
from math import sqrt

class EM():
    def __init__(self, maps=None, weights=None, depth=3):
        self.maps = maps
        self.weights = weights
        self.depth_weights = np.log(np.ones((depth,)) / depth)
        self.depth = depth
        self.post_transform = None
        self.pk_map = {}
        # self.codes = []

    def train(self, data):
        if not self.maps:
            self.maps, self.weights = self.create_initial_model()

        self.iter_once(data)

    def iter_once(self, data):
        p, pks, scalars, translations = self.e_step(data)
        self.m_step(data, p, pks, scalars, translations)


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
        transformed_data = np.repeat(transformed_data[:,:,None], len(self.codes), axis=2)

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
            # print(p.shape)
            # print(self.pk_map[i])
            pks.append(p[:, self.pk_map[i]])

        return p, pks, scalars, translations

    def m_step(self, data, p, pks, scalars, translations):
        '''

        :param data:
        :return:
        '''
        self.update_depths(p)
        self.update_weights(pks)

        inverse_map = self.post_transform.invert()
        transformed_data = np.apply_along_axis(inverse_map.apply, 1, data)
        maps = self.update_maps(transformed_data, pks, translations, scalars)
        post = self.update_post_transform(data, p, scalars, translations)


        self.maps = maps
        self.post_transform = post


    def update_depths(self, p):
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

    def update_weights(self, pks):
        '''

        :param p:
        :param pks:
        :return:
        '''
        new_weights = []
        for i in range(len(self.weights)):
            new_weights.append(np.sum(pks[i]))
        new_weights = np.array(new_weights)
        new_weights = new_weights / np.sum(new_weights)
        self.weights = new_weights

    def update_maps(self, transformed_data, pks, translations, scalars):
        '''

        :param p:
        :param pks:
        :param z:
        :param t:
        :return:
        '''
        maps = []
        for i in range(len(self.weights)):
            Z = self.create_z(i, scalars)

            Pk = pks[i]
            p_k_z = np.sum(Pk @ Z)
            T = self.create_T(translations)[:,self.pk_map[i]]
            ones = np.ones((Z.shape[1],)).T

            y_k = (1 / p_k_z) * (transformed_data.T @ Pk @ Z @ ones)
            t_k = (1 / p_k_z) * (T @ Z @ Pk.T @ np.ones((Pk.shape[0],)))

            Y_k = transformed_data.T - np.outer(y_k, np.ones((transformed_data.shape[0],)))

            T_k = T - np.outer(t_k, np.ones((T.shape[1],)))

            A = Y_k @ Pk @ Z @ T_k.T
            u, s, v_t = np.linalg.svd(A)

            # should look like 1,1,1,1,..., det(UV_T)
            rot_svd_diag = np.ones(A.shape[1])
            rot_svd_diag[-1] = np.linalg.det(u @ v_t)

            r_diag = np.diag(rot_svd_diag)

            rot = u @ r_diag @ v_t

            # solve the scalar equation
            inner = np.diag(Pk @ Z @ ones)
            a = np.einsum('ii', (Y_k @ inner @ Y_k.T)) # trace
            b = np.einsum('ii', (T_k @ Z @ Pk.T @ Y_k.T @ rot))
            # print("rot", rot.shape)
            c = -1 * transformed_data.shape[1] * p_k_z
            s_hat = self.solve_scale(a,b,c)

            t_hat = y_k - s_hat * rot @ t_k

            sim = Similitude(s_hat, rot, t_hat)
            maps.append(sim)

        return maps

    def create_z(self, i, scalars):
        '''
        Creates the Z matrix, a diagonal of scalars associated with codes that don't begin with i, given i
        and one row of the scalars matrix
        :param i:
        :param scalars:
        :return:
        '''
        diagonal = np.power(scalars[self.pk_map[i]], -2)
        return np.diag(np.diag(diagonal))

    def create_T(self, translations):
        '''

        :param translations:
        :return:
        '''
        return translations[0, :, :]

    def solve_scale(self, a, b, c):
        '''
        Solves quadratic equation for scale
        :param a:
        :param b:
        :param c:
        :return:
        '''
        bac = b**2 - 4 * a * c
        plus_sol = (-1 * b + sqrt(bac))/(2 * a)
        minus_sol = (-1 * b - sqrt(bac))/(2 * a)
        plus_sol = max(1 / plus_sol, 0)
        minus_sol = max(1 / minus_sol, 0)
        sol = max(plus_sol, minus_sol)
        assert sol > 0
        return sol

    def update_post_transform(self, data, p, scalars, translations):
        '''

        :return:
        '''
        z = np.diag(np.power(np.diag(scalars), -2))
        T = self.create_T(translations)
        ones = np.ones(z.shape[0])

        pz = 1 / np.sum(p @ z)
        # print(data.shape, p.shape, z.shape, ones.shape)
        xp = pz * (data.T @ p @ z @ ones)
        tp = pz * (T @ z @ p.T @ np.ones((p.shape[0],)))

        x_centered = data.T - np.outer(xp, np.ones((data.shape[0],))) # shape (dim , n)
        # print(x_centered.shape)
        # print(T.shape, (p @ z @ np.ones((p.shape[0], ))).shape)
        t_centered = T - np.outer(tp, np.ones((T.shape[1]),))
        print(x_centered.shape, t_centered.shape)

        # get postr transform rotation
        total_val = x_centered @ p @ z @ t_centered.T
        u, s, v_t = np.linalg.svd(total_val)
        rot_svd_diag = np.ones(u.shape[1])
        rot_svd_diag[-1] = np.linalg.det(u @ v_t)
        rot_svd_diag = np.diag(rot_svd_diag)
        post_rot = u @ rot_svd_diag @ v_t

        # post transform scalar
        a = data.T @ np.diag(p @ z @ ones) @ data
        b = T @ z @ p.T @ data @ post_rot
        c = -1 * data.shape[1] * np.sum(p)
        a = np.einsum('ii', a)
        b = np.einsum('ii', b)
        post_scalar = self.solve_scale(a,b,c)

        post_transform = xp - (post_scalar * post_rot @ tp)

        return Similitude(post_scalar, post_rot, post_transform)

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
            pre_len = len(codes)
            codes_temp = self.codes_at_depth(codons, i)
            codes += codes_temp
            if codes_temp == [[]]:
                continue
            for j in range(len(codes_temp)):
                code = codes_temp[j]

                if code[0] in pk_map:
                    pk_map[code[0]].append(pre_len + j)
                else:
                    pk_map[code[0]] = [pre_len + j]
        self.pk_map = pk_map
        # self.codes = codes
        return codes

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
            depth_log = self.depth_weights[len(code) - 1]
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
        translations = np.stack(translations, axis=2)

        # get dimensions right, want there to be (n, m) where m = len(codes)
        # codes = np.tile(codes, (data_dim,1))
        code_weights = np.tile(code_weights, (data_dim,1))
        depth_probs = np.tile(depth_probs, (data_dim,1))
        scalars = np.tile(scalars, (data_dim,1))

        return code_weights, depth_probs, scalars, translations



