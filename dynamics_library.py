import numpy as np
import types
import inspect


class PendulumCartContinuous:
    def __init__(self, m=0.2, g=9.8, l=0.3, j=0.006, b=0.1):
        self.m = m
        self.g = g
        self.l = l
        self.j = j
        self.b = b

    def __call__(self, x: np.ndarray, u: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 2
        assert u.shape[0] == 1
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        m, g, l, j, b = self.m, self.g, self.l, self.j, self.b
        x_dot[0, :] = x[1, :]
        x_dot[1, :] = m * g * l / j * np.sin(x[0, :]) - b / j * x[1, :] + 1 / j * np.cos(x[0, :]) * u[0, :]
        return x_dot


class DoubleIntegrator:
    def __init__(self):
        self.i = 1

    def __call__(self, x, u):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 2
        assert u.shape[0] == 1
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        x_dot[0, :] = x[1, :]
        x_dot[1, :] = u * self.i
        return x_dot


class SampleAndHold:
    def __init__(self, continuous_function: types.FunctionType, sample_time: float, discretization_step=-1):
        assert inspect.isclass(type(continuous_function)) or isinstance(continuous_function, types.FunctionType)
        self.continuous_function = continuous_function
        self.sample_time = sample_time
        if discretization_step == -1:
            self.discretization_step = sample_time
        else:
            self.discretization_step = discretization_step

    def __call__(self, x: np.ndarray, u: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        t = 0
        while t < self.sample_time:
            x_dot = self.continuous_function(x, u)
            x_plus = x + x_dot * self.discretization_step
            t += self.discretization_step
            x = x_plus
        return x_plus
