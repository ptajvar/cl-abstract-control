import numpy as np
from scipy.optimize import linprog
from copy import deepcopy, copy
from zonotope_lib import Box, Zonotope, size_to_box, plot_zonotope
import gurobipy as gp
from gurobipy import GRB
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

    def get_rx(self, start_set: Box) -> Zonotope:
        rx = self.A * start_set
        rx.center = rx.center + self.C
        return rx

    def get_rx_cl(self, feedback_rule, start_set: Box) -> Zonotope:
        rx = (self.A + self.B@feedback_rule) * start_set
        rx.center = self.A@start_set.center
        rx.center = rx.center + self.C
        return rx

    def get_ru(self, input: [Box, np.ndarray]) -> Zonotope:
        if isinstance(input, Box):
            return self.B * input
        elif isinstance(input, np.ndarray):
            output = self.B @ input
            output = output.reshape((len(output), 1))
            output_as_range = np.concatenate((output, output), 1)
            output_box = Box(output_as_range)
            return output_box
        elif isinstance(input, float):
            output = self.B * input
            output = output.reshape((len(output), 1))
            output_as_range = np.concatenate((output, output), 1)
            output_box = Box(output_as_range)
            return output_box

    def compute_reachable_set(self, start_set: Box, input_range: Box = None) -> Zonotope:
        reachable_set = self.get_rx(start_set)
        if input_range is not None:
            reachable_set = reachable_set + self.get_ru(input_range)
        reachable_set = reachable_set + self.W
        return reachable_set

    def compute_reachable_set_cl(self,feedback_rule, start_set: Box, input_range: Box = None) -> Zonotope:
        reachable_set = self.get_rx_cl(feedback_rule, start_set)
        if input_range is not None:
            reachable_set = reachable_set + self.get_ru(input_range)
        reachable_set = reachable_set + self.W
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
        child = StateCell(state_range, layer=self.layer + 1)
        child.refinement_list = self.refinement_list
        child.multi_step_dynamics = self.multi_step_dynamics
        self.children.append(child)

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
            return self.closed_loop_dynamics, self.input_use_range, self.feedback_control
        multisys = self.multi_step_dynamics
        assert isinstance(multisys, AffineSys)
        # error_bound = multisys.W.get_bounding_box_size()
        # t_size = target_size.get_bounding_box_size()
        # t_size[t_size < error_bound] = error_bound[t_size < error_bound]
        # target_size = 1.01*size_to_box(t_size)
        # target_size = multisys.W.get_bounding_box()
        start_size = 1.02 * target_size
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
            return self.closed_loop_dynamics, self.input_use_range, feedback_rule
        else:
            # print("no feedback!")
            self.closed_loop_dynamics = self.multi_step_dynamics
            self.input_use_range = 0.0
            return self.multi_step_dynamics, 0.0, None

    def get_controller(self, x: np.ndarray):
        assert self.contains(x)
        if self.knows_control:
            return self.control(x)
        if self.has_child():
            for child in self.children:
                if child.as_box().contains(x):
                    return child.get_controller(x)

    def get_cell_min_stage(self, x: np.ndarray) -> 'StateCell':
        current_valid = self.as_box().contains(x)
        nominee = None
        nominee_stage = np.inf
        if current_valid:
            if self.is_winning:
                nominee = self
                nominee_stage = self.stage
            if self.has_child():
                for ch in self.children:
                    assert isinstance(ch, StateCell)
                    candidate = ch.get_cell_min_stage(x)
                    if candidate is not None:
                        if candidate.stage is not None and candidate.stage < nominee_stage:
                            nominee = candidate
                            nominee_stage = candidate.stage
        return nominee

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

    def compute_multistep_affine(self, F, n_time_steps: int, input_box: Box) -> AffineSys:
        self.multi_step_dynamics = compute_multistep_affine_dynamics(F, n_time_steps, self.as_box(), input_box)
        return multistep_dynamics

def augment_dynamics(dynamics1: AffineSys, dynamics2: AffineSys):
    assert dynamics1.state_size == dynamics2.state_size and dynamics1.input_size == dynamics2.input_size
    augmented_dynamics = AffineSys(dynamics1.state_size, dynamics1.input_size)
    augmented_dynamics.A = dynamics2.A @ dynamics1.A
    augmented_dynamics.B = np.concatenate((dynamics2.A @ dynamics1.B, dynamics2.B), axis=1)
    augmented_dynamics.W = (dynamics2.A * dynamics1.W.minkowski_sum(dynamics2.W)).get_bounding_box()
    augmented_dynamics.C = dynamics2.A @ dynamics1.C + dynamics2.C
    return augmented_dynamics


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

def compute_multistep_affine_dynamics(F, n_time_steps: int, state_box: Box, input_box: Box) -> AffineSys:
    reachable_set = deepcopy(state_box)
    dynamics = get_affine_dynamics(F, reachable_set, input_box, 1000)
    multistep_dynamics = deepcopy(dynamics)
    for i in range(n_time_steps - 1):
        reachable_set = dynamics.compute_reachable_set(reachable_set, input_box)
        dynamics = get_affine_dynamics(F, reachable_set, input_box, 1000)
        multistep_dynamics = augment_dynamics(multistep_dynamics, dynamics)
    return multistep_dynamics


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


def synthesize_controller(affine_system: AffineSys, state_cell: Box, input_range: Box, target_size: Box, unbounded=False):
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
    if unbounded:
        alpha_ub = np.inf
    else:
        alpha_ub = 1
    alpha = m.addMVar(1, lb=0, ub=alpha_ub)
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

    def compute_hybridization(self, F, precision: Box, n_time_steps: int, input_box: Box):
        cell_list = [self.state_cell]
        while cell_list:
            cell = cell_list.pop(0)
            cell.dynamics = get_affine_dynamics(F, cell.as_box(), input_box, 2000)
            # cell.multi_step_dynamics = get_multistep_system(cell.dynamics, n_time_steps)
            cell.compute_multistep_affine(F, n_time_steps, input_box)
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
        if isinstance(x, np.ndarray):
            x = Box(np.array([[x[0], x[0]], [x[1], x[1]]]))
        elif isinstance(x, StateCell):
            x = x.as_box()
        assert isinstance(x, Box)
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
    while cell_list:
        # if time.time() - start_time > 10:
        #     break
        x = cell_list.pop(0)
        if x.fully_solved():
            # print("hoi")
            continue
        if np.all(x.as_box().get_bounding_box_size() < target.get_bounding_box_size() / 16):
            break
        # print(x.as_box().get_bounding_box_size())
        assert isinstance(x, StateCell)
        # if np.all(x.as_box().get_bounding_box_size() < winning_size / 4):
        #     print("not worth it")
        #     break
        parent_cell = pwa_system.get_parent_cell(x)
        affine_dynamics = parent_cell.get_multistep_dynamics()
        reachable_set = affine_dynamics.compute_reachable_set(x.as_box(), input_range)
        if not reachable_set.get_bounding_box().intersects(target):
            # print('boo')
            continue
        cl_dynamics, alpha, feedback_rule = parent_cell.get_closed_loop_dynamics(input_range, target)
        x.multi_step_dynamics = deepcopy(affine_dynamics)
        x.closed_loop_dynamics = deepcopy(cl_dynamics)
        assert isinstance(alpha, float)
        if alpha > 0.0:
            x.feedback_control = feedback_rule
        tolerance = (target.get_bounding_box_size() - cl_dynamics.compute_reachable_set(x.as_box(

        )).get_bounding_box_size()) / 2

        success, ctrl = steer(affine_system=affine_dynamics, start_state=x.as_box().center,
                              input_range=(1 - alpha) * input_range, target_state=target.center, tolerance=tolerance)
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


def compute_pre2(F, n_time_steps, X: StateCell, input_range: Box, target: Box):
    winning_size = np.zeros(target.center.shape)
    input_range_multi = input_range.get_range()
    input_range_multi = Box(np.tile(input_range_multi, (n_time_steps, 1)))
    X_new = deepcopy(X)
    cell_list = [X_new]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.fully_solved():
            continue
        if np.all(cell.as_box().get_bounding_box_size() < target.get_bounding_box_size() / 16):
            continue
        if np.all(cell.as_box().get_bounding_box_size() < winning_size / 2):
            print("not worth it")
            break
        cell.compute_multistep_affine(F, n_time_steps, input_range)
        assert isinstance(cell.multi_step_dynamics, AffineSys)
        ru = cell.multi_step_dynamics.get_ru(input_range_multi)
        ru_inv = copy(ru)
        ru_inv.center = -ru_inv.center
        pre_target_over = target.minkowski_sum(cell.multi_step_dynamics.W).minkowski_sum(ru_inv)
        if np.all(target.get_bounding_box_size()>=cell.multi_step_dynamics.W.get_bounding_box_size()):
            target_under = target.get_bounding_box()
            target_under.generators = target.generators - cell.multi_step_dynamics.W.get_bounding_box().generators
            target_under.center = target_under.center - cell.multi_step_dynamics.W.center
            pre_target_under = target_under.minkowski_sum(ru_inv)
        else:
            pre_target_under = None
        sub_list = [cell]
        while sub_list:
            sub_cell = sub_list.pop(0)
            if np.all(sub_cell.as_box().get_bounding_box_size() < winning_size / 2):
                print("not worth it")
                break
            if np.all(sub_cell.as_box().get_bounding_box_size() < target.get_bounding_box_size() / 16):
                continue
            assert isinstance(sub_cell.multi_step_dynamics, AffineSys)
            rx = sub_cell.multi_step_dynamics.get_rx(sub_cell.as_box())
            if pre_target_over.intersects(rx):
                if pre_target_over.contains(rx):
                    if pre_target_under and pre_target_under.contains(rx):
                        winning_size = np.maximum(winning_size, sub_cell.as_box().get_bounding_box_size())
                        sub_cell.is_winning = True
                    else:
                        sub_cell.split_cell()
                        cell_list = cell_list + sub_cell.children
                else:
                    sub_cell.split_cell()
                    sub_list = sub_list + sub_cell.children
            else:
                continue
    return X_new

def compute_pre_rocs(reachset_func, X: StateCell, input_list: list, check_included, check_intersects):
    X_new = deepcopy(X)
    cell_list = [X_new]
    initial_size = X_new.as_box().get_bounding_box_size()
    winning_size = np.zeros((2,1))
    while cell_list:
        cell = cell_list.pop(0)
        if cell.fully_solved():
            continue
        if np.all(cell.as_box().get_bounding_box_size() < initial_size / 16):
            continue
        intersecting = False
        for input in input_list:
            reachable_set = reachset_func(cell.as_box(), input)
            assert isinstance(reachable_set, Zonotope)
            if check_included(reachable_set.get_bounding_box()):
                winning_size = np.maximum(winning_size, cell.as_box().get_bounding_box_size())
                cell.is_winning = True
                break
            if check_intersects(reachable_set.get_bounding_box()):
                intersecting = True
        if not cell.is_winning and intersecting:
            if not cell.children:
                cell.split_cell()
            cell_list = cell_list + cell.children
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
