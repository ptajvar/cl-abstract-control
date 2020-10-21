import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from copy import copy
import gurobipy as gp

class Zonotope:
    def __init__(self, generators: np.ndarray, center: np.ndarray):
        if len(generators.shape) == 1:
            generators = generators.reshape(generators.shape[0], 1)
        if len(center.shape) == 1:
            center = center.reshape(center.shape[0], 1)
        assert generators.shape[0] == center.shape[0], "zonotope generators should have same number of columns as " \
                                                       "the zonotope center."
        self.ndim = generators.shape[0]
        self.ngen = generators.shape[1]
        self.generators = generators
        self.center = center

    def sample(self, nsamples=1) -> np.ndarray:
        rand_coefficients = np.random.rand(self.ngen, nsamples) * 2 - 1
        samples = np.matmul(self.generators, rand_coefficients) + self.center
        return samples

    def minkowski_sum(self, other_zonotope: 'Zonotope'):
        SELF = copy(self)
        SELF.ngen += other_zonotope.ngen
        SELF.generators = np.concatenate((self.generators, other_zonotope.generators), axis=1)
        return SELF

    def __rmul__(self, other):
        SELF = copy(self)
        if isinstance(other, np.ndarray):
            SELF.generators = np.matmul(other, SELF.generators)
        else:
            SELF.generators = other * SELF.generators
        return SELF

    def __add__(self, other):
        return self.minkowski_sum(other)

    def get_bounding_box_size(self):
        return np.sum(np.abs(self.generators), axis=1)

    def get_bounding_box(self):
        size_vect = self.get_bounding_box_size()
        bounding_box = size_to_box(size_vect)
        bounding_box.center = self.center
        return bounding_box

    def get_inner_box(self):
        size_vect = self.get_bounding_box_size()
        m = gp.Model()
        alpha = m.addMVar(1, lb=0, ub=1)
        beta = m.addMVar((self.ndim, self.ngen), lb=-np.inf, ub=np.inf)
        beta_abs = m.addMVar((self.ndim, self.ngen), lb=0, ub=np.inf)
        for i in range(self.ndim):
            m.addConstr(beta[i, :] <= beta_abs[i, :])
            m.addConstr(-beta[i, :] <= beta_abs[i, :])

        for i in range(self.ngen):
            vect = np.ones((self.ndim, ))
            m.addConstr(vect @ beta_abs[:, i] <= 1)

        for i in range(self.ndim):
            g = self.generators[i, :]
            for j in range(self.ndim):
                if i == j:
                    m.addConstr(g @ beta[j, :] >= size_vect[i] * alpha)
                else:
                    m.addConstr(g @ beta[j, :] == 0)

        m.setObjective(-1 * alpha)
        m.optimize()
        box = size_to_box(alpha.X * size_vect)
        box.center = self.center
        return box


    __array_priority__ = 10000

    def get_corners(self):
        corners = np.zeros((self.ndim, 2 ** self.ngen))
        corners += self.center
        for i in range(2 ** self.ngen):
            i_bin = bin(i)
            i_bin = i_bin[2:]
            for j in range(self.ngen - len(i_bin)):  # add trailing zeros
                i_bin = '0' + i_bin
            for j in range(self.ngen):
                if i_bin[j] == '0':
                    mul = -1
                else:
                    mul = 1
                corners[:, i] += mul * self.generators[:, j]
        return corners


class Box(Zonotope):
    def __init__(self, interval_ranges: np.ndarray):
        interval_lengths = np.diff(interval_ranges, axis=1).reshape(interval_ranges.shape[0], )
        generators = np.diag(interval_lengths / 2)
        center = np.mean(interval_ranges, axis=1)
        Zonotope.__init__(self, generators=generators, center=center)


def size_to_box(size_vector: np.ndarray) -> Box:
    ndim = size_vector.shape[0]
    size_vector = size_vector.reshape((ndim,1))
    box = Box(np.concatenate((-size_vector, size_vector), axis=1))
    return box


def plot_zonotope(zonotope: Zonotope, color='r--', fill=True):
    corneres = zonotope.get_corners().T
    hull = ConvexHull(corneres)
    if fill:
        plt.fill(corneres[hull.vertices, 0], corneres[hull.vertices, 1], color)
    else:
        vertices = np.concatenate((hull.vertices, hull.vertices[0:1]))
        plt.plot(corneres[vertices, 0], corneres[vertices, 1], color)
