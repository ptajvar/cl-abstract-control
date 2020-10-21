import numpy as np
import types
from zonotope_lib import NDBox, Zonotope
from pwa_lib import PieceWiseAffineFunction


def c2d(system: types.FunctionType, region: np.ndarray, lypschitz_const: float,
        time_step: float) -> PieceWiseAffineFunction:
    """
    :param system: Function pointer to the continuous system dx/dt = f(x(t),u(t))
    :param region: The region in the state*input space where we want to find the time discretization
    :param lypschitz_const: The Lypschitz constant of the system.
    :param time_step: Discretization time step
    :return: Function pointer to the discontinuous system x(k+1) = f(x(k),u(k))
    """
    assert region.shape[1] == 2, "Region should be specified as an ndim*2 matrix while there were {} columns".format(
        region.shape[1])
    region = NDBox(region)

    return system, region, lypschitz_const, time_step