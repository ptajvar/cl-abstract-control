import numpy as np
from dynamics_library import SampleAndHold, PendulumCartContinuous
from copy import copy, deepcopy
import matplotlib
import matplotlib.pyplot as plt
import zonotope_lib as ztp
import pwa_lib as pwa
import matplotlib.animation as animation

f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)

input_min = -0.5
input_max = 0.5
input_box = ztp.Box(np.array([[input_min, input_max]]))
theta_min = -2.5
theta_max = 2.5
theta_dot_min = -3.5
theta_dot_max = 3.5
state_box = ztp.Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

H = pwa.PicewiseAffineSys(state_box=state_box, input_box=input_box, refinement_list=[0])  # only refine angle
precision = ztp.size_to_box(np.array([[0.001], [0.1]]))
H.compute_hybridization(F, precision, n_time_steps=5)
print("Number of hybridization nodes: {}".format(H.size))
target = ztp.size_to_box(np.array([[0.001], [0.1]]))
pre = pwa.compute_pre(H, state_box, input_box, target)
