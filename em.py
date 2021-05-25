import numpy as np
from maps import *
from utils import identity_similitude, sphere_init
from itertools import product
from cached_property import cached_property
from math import sqrt, pow
from scipy.special import logsumexp

class EM():
    def __init__(self, maps=None, weights=None, num_components = 3, depth=3, is_centered=False, iters=100):
        self.is_centered = is_centered
        self.num_components = num_components
        self.iters = iters

        self.maps = maps
        self.weights = weights
        self.depth_weights = np.log(np.ones((depth + 1,)) / (depth + 1))
        self.depth = depth
        self.post_transform = None
        self.pk_map = {}

        self.split_variance = 0.01

    def train(self, data):
        if self.maps is None:
            self.create_initial_ifs(data.shape[1], 0.5)
        if self.is_centered:
            self.post_transform = identity_similitude(data.shape[1])
        else:
            self.post_transform = Similitude(1.0, np.eye(data.shape[1]), np.mean(data, axis=0))

        for i in range(self.iters):
            if i % 10 == 0:
                print("iter = ", i )
            self.iter_once(data)
            # for map in self.maps:
            #     print("map", map.scalar)

    def iter_once(self, data):
        p, pks, scalars, translations = self.e_step(data)
        # print("p=", p)
        self.m_step(data, p, pks, scalars, translations)


    def create_initial_ifs(self, dim, scale):
        self.maps = np.array(sphere_init(dim, self.num_components, 1, scale))
        self.weights = np.ones((self.num_components,)) / self.num_components


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
        # print("code weights", code_weights)
        # print("depth probs")
        # print("translations", translations)
        transformed_data = np.repeat(transformed_data[:,:,None], len(self.codes), axis=2)

        X_diff = transformed_data - translations
        # print("XDIFF", X_diff)
        scale_factor = -1 * h * np.log(scalars)
        # print("scalars", scalars)

        # here we want to take the dot product of each row with itself and sum along the middle axis to end with dim nxm
        dot_product = np.einsum('ijk,ijk-> ik', X_diff, X_diff)
        # print("dotproduct", dot_product)

        # -hlog(s) - (X dot X) / (2 * s^2)
        norm_log_prob = scale_factor - (np.divide(dot_product, 2 * np.square(scalars)))
        # print("norm log prob", norm_log_prob)
        p = depth_probs + code_weights + norm_log_prob
        # print("estep", p)
        # normalize rows
        sums = logsumexp(p, axis = 1)
        p = p - sums[:, np.newaxis]

        # pull out the Pk submatrices (columns associated with codes that start with k for each k)
        pks = []
        for i in range(len(self.weights)):
            # print(p.shape)
            # print(self.pk_map[i])
            pks.append(p[:, self.pk_map[i]])

        return p, pks, scalars, translations
        # return p, sums

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


        self.maps = np.array(maps)
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

        for l in range(self.depth + 1):
            Pij = p[:, d_inds[l]]
            # print("iter in update depths", l)
            # print("pij", Pij)
            # print("pij sum", np.sum(Pij))
            logsum = logsumexp(Pij)
            # print(logsum)
            self.depth_weights[l] = logsumexp([logsum, self.depth_weights[l]])
            # print("depth weights", self.depth_weights)

        tot_sum = logsumexp(self.depth_weights)
        # print("tot", tot_sum)
        # print("before", self.depth_weights)
        self.depth_weights = self.depth_weights - tot_sum
        # print("depth weights", self.depth_weights)

    def update_weights(self, pks):
        '''

        :param p:
        :param pks:
        :return:
        '''
        new_weights = []
        for i in range(len(self.weights)):
            new_weights.append(logsumexp(pks[i]))
        new_weights = np.array(new_weights)
        new_weights = new_weights - logsumexp(new_weights)
        self.weights = np.exp(new_weights)
        # print("new weights", self.weights)

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
            Z = self.create_z(scalars)
            # print("z shape", Z.shape)
            Pk = pks[i]
            # print("pk", Pk, Pk.shape)
            # print("Z", Z, Z.shape)
            pkz_matrix = Pk + Z
            # print("final", pkz_matrix)
            pkz_sum = logsumexp(pkz_matrix)
            normed_pkz = pkz_matrix - pkz_sum
            normed_pkz= np.exp(normed_pkz)

            pkz_matrix = np.exp(pkz_matrix)
            # print("normed", PKZ)
            # print("normed_sum", np.sum(PKZ))
            # p_k_z = np.exp(p_k_z)

            T = self.create_T(translations)
            T = T[:, : T.shape[1] - int(pow(self.num_components, self.depth))]
            ones = np.ones((pkz_matrix.shape[1],)).T

            # y_k = (1 / p_k_z) * (transformed_data.T @ PKZ @ ones)
            # t_k = (1 / p_k_z) * (T @ PKZ.T @ np.ones((Pk.shape[0],)))
            # print("weights", normed_pkz @ ones)
            y_k = transformed_data.T @ (normed_pkz @ ones)
            # print(T.shape, normed_pkz.shape, pkz_matrix.shape)
            temp = np.ones((pkz_matrix.shape[0],)) @ normed_pkz
            # print(temp.shape)
            t_k = T @ temp
            # print("mean y", y_k)


            Y_k = transformed_data.T - np.outer(y_k, np.ones((transformed_data.shape[0],)))

            T_k = T - np.outer(t_k, np.ones((T.shape[1],)))

            # print(Y_k, "ycenter")

            A = Y_k @ normed_pkz @ T_k.T
            # print(A)
            u, s, v_t = np.linalg.svd(A)

            # should look like 1,1,1,1,..., det(UV_T)
            rot_svd_diag = np.ones(A.shape[1])
            rot_svd_diag[-1] = np.linalg.det(u @ v_t)
            # print("diag",rot_svd_diag)

            r_diag = np.diag(rot_svd_diag)

            rot = u @ r_diag @ v_t
            if np.isnan(rot).any():
                print("Rotation matrix contains null value")
                maps.append(None)
                continue
            # solve the scalar equation

            # inner = np.diag(pkz_matrix @ ones)
            weighting = np.einsum('ij, ij-> j', Y_k, Y_k)
            vect = pkz_matrix @ ones
            # print(vect, "weights")
            a = np.sum(np.multiply(vect, weighting)) # trace
            b = -1 * np.trace(T_k @ pkz_matrix.T @ Y_k.T @ rot)
            # print("rot", rot.shape)
            c = -1 * transformed_data.shape[1] * np.exp(logsumexp(Pk))
            # print(a,b,c)
            s_hat = self.solve_scale(a,b,c)
            # print("component= ", i, s_hat)
            if s_hat is None or s_hat == np.infty:  # if singularity, then we need to split the remaining maps
                maps.append(None)
                continue

            t_hat = y_k - s_hat * rot @ t_k

            sim = Similitude(s_hat, rot, t_hat)
            maps.append(sim)

        final_maps, final_weights = self.split_maps(maps, self.weights)
        self.weights = final_weights
        return final_maps

    def split_maps(self, maps, weights):
        '''

        :param maps:
        :return:
        '''
        null_inds = []
        not_null = []
        for i in range(len(maps)):
            if maps[i] is None:
                null_inds.append(i)
            else:
                not_null.append(i)
        if not not_null:
            raise Exception("All maps in this IFS are null")

        if not null_inds:
            return maps, weights

        # random choice of a not null map to split
        choice_ind = np.random.choice(not_null)
        chosen_map = maps[choice_ind]
        split_weight = weights[choice_ind] / len(null_inds)
        null_inds.append(choice_ind)

        for null_val in null_inds:
            base_trans = chosen_map.t
            rand = np.random.sample((base_trans.shape[0],)) * self.split_variance
            new_trans = base_trans + rand
            sim = Similitude(chosen_map.scalar, chosen_map.rotation, new_trans)
            maps[null_val] = sim
            weights[null_val] = split_weight

        return maps, weights

    def create_z(self, scalars):
        '''
        Creates a vector of the Z matrix in log space, associated with codes that don't begin with i, given i
        and one row of the scalars matrix
        :param i:
        :param scalars:
        :return:
        '''
        # diagonal = np.power(scalars[self.pk_map[i]], -2)
        # return np.diag(np.diag(diagonal))
        # print(scalars[0, self.pk_map[i]].shape)
        # print(int(pow(self.num_components, self.depth)), "num")
        return -2 * np.log(scalars[0,: scalars.shape[1] - int(pow(self.num_components, self.depth))])

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
        # print("det", bac)
        if bac < 0:
            return None
        plus_sol = (-1 * b + sqrt(bac))/(2 * a)
        minus_sol = (-1 * b - sqrt(bac))/(2 * a)
        # print("before solutions", minus_sol, plus_sol)
        if plus_sol == 0:
            plus_sol = np.infty
        else:
            plus_sol = 1 / plus_sol

        if minus_sol == 0:
            minus_sol = np.infty
        else:
            minus_sol = 1 / minus_sol
        # print(minus_sol, plus_sol, "solutions")

        # plus_sol = max(plus_sol, 0)
        # minus_sol = max(minus_sol, 0)
        # sol = max(plus_sol, minus_sol)
        if minus_sol > 0 and plus_sol > 0:
            return max(minus_sol, plus_sol)

        if minus_sol > 0 :
            return minus_sol

        if plus_sol > 0:
            return plus_sol
        # if sol <= 0:
        #     print("found solution less than zero, adding null")
        #     sol = None
        print("No solution found adding null")
        return None

    def update_post_transform(self, data, p, scalars, translations):
        '''

        :return:
        '''
        z = -2 * np.log(scalars)
        T = self.create_T(translations)
        # ones = np.ones(z.shape[0])

        pz_matrix = p + z
        # print("final", pkz_matrix)
        pz_sum = logsumexp(pz_matrix)
        normed_pz = pz_matrix - pz_sum
        normed_pz = np.exp(normed_pz)

        pz_matrix = np.exp(pz_matrix)
        ones = np.ones((pz_matrix.shape[1]))

        # pz = 1 / np.sum(p @ z)
        # print(data.shape, p.shape, z.shape, ones.shape)
        # print(data.shape, normed_pz.shape, ones.shape)
        xp = data.T @ (normed_pz @ ones)
        temp = np.ones((pz_matrix.shape[0],)) @ normed_pz
        tp = T @ temp

        x_centered = data.T - np.outer(xp, np.ones((data.shape[0],))) # shape (dim , n)
        # print(x_centered.shape)
        # print(T.shape, (p @ z @ np.ones((p.shape[0], ))).shape)
        t_centered = T - np.outer(tp, np.ones((T.shape[1]),))
        # print(x_centered.shape, t_centered.shape)

        # get postr transform rotation
        total_val = x_centered @ normed_pz @ t_centered.T
        u, s, v_t = np.linalg.svd(total_val)
        rot_svd_diag = np.ones(u.shape[1])
        rot_svd_diag[-1] = np.linalg.det(u @ v_t)
        rot_svd_diag = np.diag(rot_svd_diag)
        post_rot = u @ rot_svd_diag @ v_t

        if np.isnan(post_rot).any():
            print("Rotation matrix in post transform contains null value")
            return None

        # post transform scalar
        weighting = np.einsum('ij, ij-> j', x_centered, x_centered)
        vect = pz_matrix @ ones
        a = np.sum(np.multiply(vect, weighting))
        b = t_centered @ pz_matrix.T @ x_centered.T @ post_rot
        c = -1 * data.shape[1] * logsumexp(p)
        # a = np.einsum('ii', a)
        b = np.trace(b)
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
        for i in range(self.depth + 1):
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
        translations = np.stack(translations, axis=2)

        # get dimensions right, want there to be (n, m) where m = len(codes)
        # codes = np.tile(codes, (data_dim,1))
        code_weights = np.tile(code_weights, (data_dim,1))
        depth_probs = np.tile(depth_probs, (data_dim,1))
        scalars = np.tile(scalars, (data_dim,1))

        return code_weights, depth_probs, scalars, translations



