import numpy as np

class AffineFunction:
    def __init__(self, linear_term: np.ndarray, constant_term: np.ndarray, disturbance_term: np.ndarray):
        assert linear_term.shape()[1] == constant_term.shape()[1], "Column size mismatch: The linear term has {} " \
                                                                   "columns while the constant term has {} " \
                                                                   "columns.".format(linear_term.shape()[1],
                                                                                     constant_term.shape()[1])
        if
        self.A = linear_term
        self.b = constant_term
        self.W = disturbance_term
    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if "option" in kwargs and kwargs["option"] == "random":
            return self.A * x + self.b + np.random
        return self.A*x + self.b


class PiecewiseAffineNode:
    def __init__(self, node_id: int, validity_range: np.ndarray):
        self.id = node_id
        self.validity_range = validity_range
        self.function = AffineFunction(0, 0, 0)
        self.has_child = False
        self.children = []

    def add_child(self, child_id):
        self.has_child = True
        self.children.append(child_id)


class PieceWiseAffineFunction:
    def __init__(self, validity_range: np.ndarray):
        self.root = PiecewiseAffineNode(node_id=0, validity_range=validity_range)
        self.nodes = {0:self.root}
    def __call__(self, *args, **kwargs):


