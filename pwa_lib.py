import numpy as np
from scipy.optimize import linprog
from copy import deepcopy
from zonotope_lib import Box, Zonotope, size_to_box
import gurobipy as gp


class Affine:
    """
    An error tolerant affine function of x: y \in A * [x;1] \osum W
    """

    def __init__(self, input_size, output_size):
        self.coefficient_mat = np.ndarray((output_size, input_size))  # A
        self.error_term = np.ndarray((output_size,))  # W

    def __call__(self, input_data):
        n_datapoints = input_data.shape[1]
        input_data = np.concatenate((input_data, np.ones((1, n_datapoints))), axis=0)
        y = np.matmul(self.coefficient_mat, input_data)
        return y


class AffineSys:
    """
    An error tolerant affine system: y \in A * [x] + B * u + C \osum W
    """

    def __init__(self, state_size=0, input_size=0):
        self.state_size = state_size
        self.input_size = input_size
        self.A = np.ndarray((state_size, state_size))  # A
        self.B = np.ndarray((state_size, input_size))  # B
        self.C = np.ndarray((state_size, 1))  # C
        self.W = size_to_box(np.ndarray((state_size,)))  # W

    def init_from_affine_func(self, affine_func, state_index_list):
        coefficient_mat = affine_func.coefficient_mat
        coefficient_ncolumns = coefficient_mat.shape[1]
        input_list_index = list(set(range(coefficient_ncolumns - 1)) - set(state_index_list))
        input_list_index = np.array(input_list_index)
        state_index_list = np.array(state_index_list)
        self.state_size = len(state_index_list)
        self.input_size = coefficient_ncolumns - self.state_size - 1
        self.A = coefficient_mat[:, state_index_list]
        self.B = coefficient_mat[:, input_list_index]
        self.C = coefficient_mat[:, coefficient_ncolumns - 1:coefficient_ncolumns]
        self.W = size_to_box(affine_func.error_term.reshape(affine_func.error_term.size, 1))

    def __call__(self, current_state, current_input=None):
        n_datapoints = current_input.shape[1]
        if current_input is None:
            y = np.matmul(self.A, current_state) + np.matmul(self.C, np.ones(self.state_size))
        else:
            y = np.matmul(self.A, current_state) + np.matmul(self.B, current_input) + np.matmul(self.C,
                                                                                                np.ones(
                                                                                                    self.state_size))
        return y

    def get_closed_loop(self, feedback_matrix):
        SELF = deepcopy(self)
        SELF.A = SELF.A + np.matmul(SELF.B, feedback_matrix)
        return SELF

    def compute_reachable_set(self, start_set: Box):
        reachable_set = Zonotope(np.zeros((start_set.ndim, 0)), np.zeros((start_set.ndim, 1)))
        reachable_set.generators = np.matmul(self.A, start_set.generators)
        reachable_set.center = np.matmul(self.A, start_set.center) + self.C
        # reachable_set.generators = np.concatenate((reachable_set.generators, self.error_term), axis=1)
        reachable_set.ngen = reachable_set.generators.shape[1]
        return reachable_set


def fit_affine_function(input_data, output_data, solver='gurobi'):
    """
    Fit an affine function to I/O data presented as matrices (ndarrays).
    Args:
        input_data: ni*m matrix where ni is the size of each datapoint and m is the number of datapoints.
        output_data: no*m matrix where no is the size of each datapoint.
    Returns:
        affine: an affine function that is fit to the data. affine.coefficient_mat should be an no*(ni+1) matrix and
        affine.error_term should be an no*1 matrix.
    """
    # variables are A and w, coefficients are input_data and output_data
    # aug_input_data = [input_data; 1]

    n_datapoints = input_data.shape[1]
    input_data = np.concatenate((input_data, np.ones((1, n_datapoints))), axis=0)
    input_size = input_data.shape[0]
    output_size = output_data.shape[0]
    affine_mat_size = output_size * input_size
    error_vec_size = output_size
    solution_size = affine_mat_size + error_vec_size

    # cost function: sum w (L1 norm)
    cost_func_affine = np.zeros(affine_mat_size)
    cost_func_error = np.ones(error_vec_size)
    cost_func = np.concatenate((cost_func_affine, cost_func_error))

    b_ub = np.zeros((output_size * n_datapoints * 2, 1))
    A_ub = np.zeros((output_size * n_datapoints * 2, solution_size))
    for i in range(n_datapoints):  # A * aug_input_data - w < output_data
        for j in range(output_size):
            row_i = i * output_size + j
            b_ub[row_i] = output_data[j, i]
            A_ub[row_i][-error_vec_size + j] = -1
            A_ub[row_i][j * input_size:(j + 1) * input_size] = input_data[:, i].T

    for i in range(n_datapoints):  # - A * aug_input_data - w < - output_data
        for j in range(output_size):
            row_i = i * output_size + j + output_size * n_datapoints
            b_ub[row_i] = -output_data[j, i]
            A_ub[row_i][-error_vec_size + j] = -1
            A_ub[row_i][j * input_size:(j + 1) * input_size] = -input_data[:, i].T
    affine_func = Affine(input_size, output_size)

    if solver == 'gurobi':
        m = gp.Model()
        x = m.addMVar((cost_func.size,), lb=-np.inf)
        m.setObjective(cost_func @ x)
        m.addConstr(A_ub @ x <= b_ub.reshape(len(b_ub)))
        m.optimize()
        for i in range(output_size):
            affine_func.coefficient_mat[i][:] = x.X[i * input_size:(i + 1) * input_size]
        affine_func.error_term = x.X[-error_vec_size:]
    else:
        options = {"disp": True, "maxiter": 50000, "tol": 1e-8}
        solution = linprog(c=cost_func.tolist(), A_ub=A_ub.tolist(), b_ub=b_ub.tolist(), bounds=(None, None),
                           options=options)
        for i in range(output_size):
            affine_func.coefficient_mat[i][:] = solution.x[i * input_size:(i + 1) * input_size]
        affine_func.error_term = solution.x[-error_vec_size:]
    return affine_func


def get_multistep_system(affine_system: AffineSys, n_time_steps) -> AffineSys:
    multi_step_system = deepcopy(affine_system)
    W0 = affine_system.W
    W = deepcopy(affine_system.W)
    for i in range(n_time_steps):
        multi_step_system.A = np.matmul(affine_system.A, multi_step_system.A)
        multi_step_system.B = np.concatenate((np.matmul(affine_system.A, multi_step_system.B), affine_system.B), axis=1)
        multi_step_system.C = np.matmul(affine_system.A, multi_step_system.C) + affine_system.C
        W = affine_system.A * W + W0
    multi_step_system.W = W
    return multi_step_system


def get_affine_dynamics(F, state_region: Box, input_region: Box, n_sample=1000):
    samples_x = state_region.sample(n_sample)
    samples_u = input_region.sample(n_sample)
    samples_in = np.concatenate((samples_x, samples_u), axis=0)
    samples_x_plus = F(samples_x, samples_u)
    affine_func = fit_affine_function(input_data=samples_in, output_data=samples_x_plus)
    affine_system = AffineSys()
    affine_system.init_from_affine_func(affine_func=affine_func, state_index_list=list(range(0, state_region.ndim)))
    return affine_system


def can_reach(affine_system: AffineSys, begin: Box, target: Box):
    begin_center = begin.center
    target_center = target.center
    u = np.matmul(np.linalg.pinv(affine_system.B), target_center - affine_system.C - affine_system.A * begin_center)
    if np.all(u < 10) and np.all(u > -10):
        return True
    return False


def synthesize_controller(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box):
    state_dim = state_cell.ndim
    input_dim = input_range.ndim
    n_step_input_dim = affine_system.B.shape[1]
    state_bounds = np.sum(np.abs(state_cell.generators), axis=1)
    input_bounds = np.sum(np.abs(input_range.generators), axis=1)
    target_bounds = np.sum(np.abs(target_size.generators), axis=1) - affine_system.W.get_bounding_box_size()
    m = gp.Model()
    alpha = m.addMVar(1, lb=0, ub=1)
    c = m.addMVar((n_step_input_dim, state_dim), lb=-np.inf)
    c_abs = m.addMVar((n_step_input_dim, state_dim), lb=0)
    for i in range(n_step_input_dim):
        i_ind = i % input_dim
        m.addConstr(c[i, :] <= c_abs[i, :])
        m.addConstr(-c[i, :] <= c_abs[i, :])
        m.addConstr(state_bounds @ c_abs[i, :] <= input_bounds[i_ind] * alpha)

    A_plus_Bc_abs = m.addMVar((state_dim, state_dim), lb=0)
    for i in range(state_dim):
        for j in range(state_dim):
            m.addConstr(affine_system.B[i, :] @ c[:, j] + affine_system.A[i, j] <= A_plus_Bc_abs[i, j])
            m.addConstr(-affine_system.B[i, :] @ c[:, j] - affine_system.A[i, j] <= A_plus_Bc_abs[i, j])

    for i in range(state_dim):
        m.addConstr(state_bounds @ A_plus_Bc_abs[i, :] <= target_bounds[i])
    m.setObjective(1 * alpha)
    m.optimize()
    feedback_rule = c.X
    return feedback_rule, alpha.X


def synthesize_controller(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box):
    state_dim = state_cell.ndim
    input_dim = input_range.ndim
    n_step_input_dim = affine_system.B.shape[1]
    state_bounds = np.sum(np.abs(state_cell.generators), axis=1)
    input_bounds = np.sum(np.abs(input_range.generators), axis=1)
    target_bounds = np.sum(np.abs(target_size.generators), axis=1) - affine_system.W.get_bounding_box_size()
    m = gp.Model()
    alpha = m.addMVar(1, lb=0, ub=1)
    c = m.addMVar((n_step_input_dim, state_dim), lb=-np.inf)
    c_abs = m.addMVar((n_step_input_dim, state_dim), lb=0)
    for i in range(n_step_input_dim):
        i_ind = i % input_dim
        m.addConstr(c[i, :] <= c_abs[i, :])
        m.addConstr(-c[i, :] <= c_abs[i, :])
        m.addConstr(state_bounds @ c_abs[i, :] <= input_bounds[i_ind] * alpha)

    A_plus_Bc_abs = m.addMVar((state_dim, state_dim), lb=0)
    for i in range(state_dim):
        for j in range(state_dim):
            m.addConstr(affine_system.B[i, :] @ c[:, j] + affine_system.A[i, j] <= A_plus_Bc_abs[i, j])
            m.addConstr(-affine_system.B[i, :] @ c[:, j] - affine_system.A[i, j] <= A_plus_Bc_abs[i, j])

    for i in range(state_dim):
        m.addConstr(state_bounds @ A_plus_Bc_abs[i, :] <= target_bounds[i])
    m.setObjective(1 * alpha)
    m.optimize()
    feedback_rule = c.X
    return feedback_rule, alpha.X
