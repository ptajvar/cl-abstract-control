import numpy as np
from scipy.optimize import linprog
from copy import deepcopy, copy
from zonotope_lib import Box, Zonotope, size_to_box, plot_zonotope
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time


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

    def compute_reachable_set(self, start_set: Box, input_range: Box = None) -> Zonotope:
        reachable_set = Zonotope(np.zeros((start_set.ndim, 0)), np.zeros((start_set.ndim, 1)))
        reachable_set.generators = np.matmul(self.A, start_set.generators)
        reachable_set.generators = np.concatenate((reachable_set.generators, self.W.generators), axis=1)
        reachable_set.center = np.matmul(self.A, start_set.center) + self.C + self.W.center
        # reachable_set.generators = np.concatenate((reachable_set.generators, self.error_term), axis=1)
        reachable_set.ngen = reachable_set.generators.shape[1]
        if input_range is not None:
            n_steps = int(self.B.shape[1] / input_range.generators.shape[0])
            in_gen = np.tile(input_range.generators, (n_steps, 1))
            reachable_set.generators = np.concatenate((reachable_set.generators, np.matmul(self.B, in_gen)), axis=1)
        return reachable_set


class StateCell:
    def __init__(self, range_matrix: np.ndarray, layer=0):
        self.range = range_matrix
        self.children = []
        self.layer = layer
        self.ndim = range_matrix.shape[0]
        self.refinement_list = list(range(0, self.ndim))
        self.is_winning = False
        self.dynamics = None
        self.multi_step_dynamics = None
        self.feedback_control = None
        self.feedfwd_control = None
        self.input_use_range = None
        self.closed_loop_dynamics = None
        self.stage = None

    def has_child(self):
        if self.children:
            return True
        return False

    def split_cell(self):
        rl_length = len(self.refinement_list)
        split_dim = self.refinement_list[self.layer % rl_length]
        half_point = np.mean(self.range[split_dim, :])

        child1_range = copy(self.range)
        child2_range = copy(self.range)

        child1_range[split_dim, 1] = half_point
        child2_range[split_dim, 0] = half_point

        self.add_child(child1_range)
        self.add_child(child2_range)

    def add_child(self, state_range):
        self.children.append(StateCell(state_range, layer=self.layer + 1))
        self.children[-1].refinement_list = self.refinement_list

    def fully_solved(self):
        if self.is_winning:
            return True
        elif self.children:
            for child in self.children:
                if not child.fully_solved():
                    return False
            return True
        else:
            return False

    def set_feedback_controller(self, controller):
        self.feedback_control = controller

    def get_closed_loop_dynamics(self, input_box=None, target_size=None):
        if self.closed_loop_dynamics is not None:
            return self.closed_loop_dynamics, self.input_use_range
        multisys = self.multi_step_dynamics
        assert isinstance(multisys, AffineSys)
        # error_bound = multisys.W.get_bounding_box_size()
        # t_size = target_size.get_bounding_box_size()
        # t_size[t_size < error_bound] = error_bound[t_size < error_bound]
        # target_size = 1.01*size_to_box(t_size)
        target_size = multisys.W.get_bounding_box()
        start_size = 1.51 * target_size
        # start_size.generators[1, 1] += 1.0
        success, feedback_rule, alpha, closed_loop_system = synthesize_controller(affine_system=multisys,
                                                                                  state_cell=start_size,
                                                                                  input_range=input_box,
                                                                                  target_size=1.01 * target_size)

        if success:
            self.feedback_control = feedback_rule
            self.closed_loop_dynamics = closed_loop_system
            self.input_use_range = alpha
            # print(alpha)
            return self.closed_loop_dynamics, self.input_use_range
        else:
            # print("no feedback!")
            self.closed_loop_dynamics = self.multi_step_dynamics
            self.input_use_range = 0.0
            return self.multi_step_dynamics, 0.0

    def get_controller(self, x: np.ndarray):
        assert self.contains(x)
        if self.knows_control:
            return self.control(x)
        if self.has_child():
            for child in self.children:
                if child.as_box().contains(x):
                    return child.get_controller(x)

    def as_box(self):
        b = Box(self.range)
        return b

    def get_multistep_dynamics(self) -> AffineSys:
        return self.multi_step_dynamics

    def get_bare_copy(self):
        bare_copy = StateCell(self.range, self.layer)
        if self.has_child():
            for i in self.children:
                bare_copy.children.append(i.get_bare_copy())
        return bare_copy


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
        m.setParam('OutputFlag', 0)
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
    W0 = deepcopy(affine_system.W)
    W = deepcopy(affine_system.W)
    for i in range(n_time_steps - 1):
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


# def synthesize_controller(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box):
#     state_dim = state_cell.ndim
#     input_dim = input_range.ndim
#     n_step_input_dim = affine_system.B.shape[1]
#     state_bounds = np.sum(np.abs(state_cell.generators), axis=1)
#     input_bounds = np.sum(np.abs(input_range.generators), axis=1)
#     target_bounds = np.sum(np.abs(target_size.generators), axis=1) - affine_system.W.get_bounding_box_size()
#     m = gp.Model()
#     alpha = m.addMVar(1, lb=0, ub=1)
#     c = m.addMVar((n_step_input_dim, state_dim), lb=-np.inf)
#     c_abs = m.addMVar((n_step_input_dim, state_dim), lb=0)
#     for i in range(n_step_input_dim):
#         i_ind = i % input_dim
#         m.addConstr(c[i, :] <= c_abs[i, :])
#         m.addConstr(-c[i, :] <= c_abs[i, :])
#         m.addConstr(state_bounds @ c_abs[i, :] <= input_bounds[i_ind] * alpha)
#
#     A_plus_Bc_abs = m.addMVar((state_dim, state_dim), lb=0)
#     for i in range(state_dim):
#         for j in range(state_dim):
#             m.addConstr(affine_system.B[i, :] @ c[:, j] + affine_system.A[i, j] <= A_plus_Bc_abs[i, j])
#             m.addConstr(-affine_system.B[i, :] @ c[:, j] - affine_system.A[i, j] <= A_plus_Bc_abs[i, j])
#
#     for i in range(state_dim):
#         m.addConstr(state_bounds @ A_plus_Bc_abs[i, :] <= target_bounds[i])
#     m.setObjective(1 * alpha)
#     m.optimize()
#     feedback_rule = c.X
#     return feedback_rule, alpha.X


def synthesize_controller(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box):
    state_dim = state_cell.ndim
    input_dim = input_range.ndim
    n_step_input_dim = affine_system.B.shape[1]
    state_bounds = np.sum(np.abs(state_cell.generators), axis=1)
    input_bounds = np.sum(np.abs(input_range.generators), axis=1)
    target_bounds = np.sum(np.abs(target_size.generators), axis=1).reshape(state_dim, 1) - \
                    affine_system.W.get_bounding_box_size() / 2
    if np.any(target_bounds <= 0):
        success = False
        return success, None, None, None
    m = gp.Model()
    m.setParam('OutputFlag', 0)
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
    a = GRB.OPTIMAL
    b = m.getAttr('Status')
    if m.getAttr('Status') == GRB.OPTIMAL:
        success = True
        feedback_rule = c.X
        closed_loop_system = AffineSys(affine_system.state_size, n_step_input_dim)
        closed_loop_system.A = affine_system.A + affine_system.B @ feedback_rule
        closed_loop_system.B = affine_system.B
        closed_loop_system.C = affine_system.C
        closed_loop_system.W = affine_system.W
        return success, feedback_rule, float(alpha.X), closed_loop_system
    else:
        success = False
        return success, None, None, None


def steer(affine_system: AffineSys, start_state: np.ndarray, target_state: np.ndarray, input_range: Box,
          tolerance=None):
    if tolerance is None:
        tolerance = np.zeros((affine_system.state_size, 1))
    if np.any(tolerance < 0):
        return False, None
    input_bounds = np.sum(np.abs(input_range.generators), axis=1)
    n_step_input_dim = affine_system.B.shape[1]
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    input_less_state = affine_system.A @ start_state + affine_system.C + affine_system.W.center
    c = m.addMVar((n_step_input_dim,), lb=-np.inf, ub=np.inf)
    c_abs = m.addMVar((n_step_input_dim,), lb=0, ub=np.inf)
    for i in range(n_step_input_dim):
        iid = i % input_bounds.size
        m.addConstr(c[i] - input_range.center[iid] <= c_abs[i])
        m.addConstr(-c[i] + input_range.center[iid] <= c_abs[i])
        m.addConstr(c_abs[i] <= input_bounds[iid])
    for i in range(affine_system.state_size):
        m.addConstr(affine_system.B[i, :] @ c + input_less_state[i] - target_state[i] <= tolerance[i])
        m.addConstr(-(affine_system.B[i, :] @ c + input_less_state[i] - target_state[i]) <= tolerance[i])
    m.optimize()
    if m.getAttr('Status') == GRB.OPTIMAL:
        # print("--------------------------------------------------------------------------")
        success = True
        return success, c.X
    else:
        success = False
        return success, None


def synthesize_controller_cl(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box):
    state_dim = state_cell.ndim
    input_dim = input_range.ndim
    n_step_input_dim = affine_system.B.shape[1]
    state_bounds = np.sum(np.abs(state_cell.generators), axis=1)
    input_bounds = np.sum(np.abs(input_range.generators), axis=1)
    target_bounds = np.sum(np.abs(target_size.generators), axis=1)
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    alpha = m.addMVar(1, lb=0, ub=1)
    c = m.addMVar((input_dim, state_dim), lb=-np.inf)
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
    m.printStats()
    feedback_rule = c.X
    return feedback_rule, float(alpha.X)


class PiecewiseAffineSys:
    def __init__(self, input_box: Box, state_box: Box, refinement_list: list):
        """
        :param input_box: allowed inputs
        :param state_box: The state space where the hybridization is consturcteed
        :param refinement_list: List of the state dimensions that can be refined during hybridization
        """
        self.state_cell = StateCell(state_box.get_range())
        self.input_box = input_box
        self.state_cell.refinement_list = refinement_list
        self.size = 1

    def compute_hybridization(self, F, precision: Box, n_time_steps, input_box: Box):
        cell_list = [self.state_cell]
        while cell_list:
            cell = cell_list.pop(0)
            cell.dynamics = get_affine_dynamics(F, cell.as_box(), input_box, 2000)
            cell.multi_step_dynamics = get_multistep_system(cell.dynamics, n_time_steps)
            if not precision.contains(cell.multi_step_dynamics.W):
                cell.split_cell()
                cell_list = cell_list + cell.children
                self.size += len(cell.children)

    def __call__(self, x, u):
        if isinstance(x, np.ndarray):
            x = x.reshape(x.size, 1)
            x = Box(np.concatenate((x, x), axis=1))
        if isinstance(u, np.ndarray):
            u = u.reshape(u.size, 1)
            u = Box(np.concatenate((u, u), axis=1))
        assert isinstance(x, Zonotope)
        assert isinstance(u, Zonotope)
        self.compute_reachable_set(x, u)

    def get_parent_cell(self, x: Box) -> StateCell:
        if isinstance(x, StateCell):
            x = x.as_box()
        if not self.state_cell.as_box().contains(x):
            print("State out of range")
            return None
        smallest_cell = self.state_cell
        while smallest_cell.has_child():
            children = smallest_cell.children
            found_valid = False
            for ch in children:
                if ch.as_box().contains(x):
                    smallest_cell = ch
                    found_valid = True
                    break
            if not found_valid:
                break
        return smallest_cell


def compute_pre(pwa_system: PiecewiseAffineSys, X: StateCell, input_range: Box, target: Box):
    winning_size = np.zeros(target.center.shape)
    start_time = time.time()
    X_new = deepcopy(X)
    cell_list = [X_new]
    pre_set = []
    while cell_list:
        # if time.time() - start_time > 10:
        #     break
        x = cell_list.pop(0)
        if x.fully_solved():
            continue
            print("hoi")
        if np.all(x.as_box().get_bounding_box_size() < target.get_bounding_box_size() / 8):
            break
        # print(x.as_box().get_bounding_box_size())
        assert isinstance(x, StateCell)
        if np.all(x.as_box().get_bounding_box_size() < winning_size/4):
            print("not worth it")
            break
        parent_cell = pwa_system.get_parent_cell(x)
        affine_dynamics = parent_cell.get_multistep_dynamics()
        reachable_set = affine_dynamics.compute_reachable_set(x.as_box(), input_range)
        if not reachable_set.get_bounding_box().intersects(target):
            # print('boo')
            continue
        cl_dynamics, alpha = parent_cell.get_closed_loop_dynamics(input_range, target)

        assert isinstance(alpha, float)
        # if alpha > 0.0:
        tolerance = target.get_bounding_box_size() - cl_dynamics.compute_reachable_set(x.as_box(

        )).get_bounding_box_size()
        success, ctrl = steer(affine_system=cl_dynamics, start_state=x.as_box().center, input_range=(
                                                                                                            1 - alpha) * input_range,
                              target_state=target.center,
                              tolerance=tolerance)
        # else:
        #     print("no feedback")
        #     success = False

        if success:
            winning_size = np.maximum(winning_size, x.as_box().get_bounding_box_size())
            x.is_winning = True
            x.feedfwd_control = ctrl
        else:
            if x.has_child():
                cell_list = cell_list + x.children
            else:
                x.split_cell()
                cell_list = cell_list + x.children
    return X_new


def get_attraction(F: AffineSys, target: Box, input_region: Box, n_steps: int):
    state_region = deepcopy(target)
    F_affine = get_affine_dynamics(F, state_region, input_region)
    F_affine_multi = get_multistep_system(F_affine, n_steps)
    while np.all(F_affine_multi.W.get_bounding_box_size() < target.get_bounding_box_size()):
        F_affine_multi_old = deepcopy(F_affine_multi)
        state_region_old = deepcopy(state_region)
        state_region = 1.1 * state_region
        F_affine = get_affine_dynamics(F, state_region, input_region)
        F_affine_multi = get_multistep_system(F_affine, n_steps)
    F_affine_multi = deepcopy(F_affine_multi_old)
    state_region = deepcopy(state_region_old)
    success, feedback_rule, alpha, closed_loop_system = synthesize_controller(F_affine_multi, 1.01 * state_region,
                                                                              input_region, state_region)
    return state_region, feedback_rule
