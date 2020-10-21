import numpy as np
from dynamics_library import SampleAndHold, PendulumCartContinuous
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
import zonotope_lib as ztp
import pwa_lib as pwa
import matplotlib.animation as animation

f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)
run_time = 0.12
state_region = ztp.Box(np.array([[-1, +1], [-10, 10]]))
input_region = ztp.Box(np.array([[-10, 10]]))
F_linear = pwa.get_affine_dynamics(F=F, state_region=state_region, input_region=input_region, n_sample=200)
n_time_steps = 12
F_linear_multistep = pwa.get_multistep_system(affine_system=F_linear, n_time_steps=n_time_steps)

min_cell_size = F_linear_multistep.W.get_bounding_box()

feed_back_law, alpha = pwa.synthesize_controller(F_linear_multistep, 1.1*min_cell_size, input_region, 1.1*min_cell_size)


# ztp.plot_zonotope(ztp.Zonotope(10*F_linear_multistep.B, np.array([[0], [0]])).get_bounding_box(), color='b')
# ztp.plot_zonotope(ztp.Zonotope(F_linear_multistep.A, np.array([[0], [0]])), color='b')

ztp.plot_zonotope(ztp.Zonotope(10*F_linear_multistep.B, np.array([[0], [0]])))
ztp.plot_zonotope(ztp.Zonotope(10*F_linear_multistep.B, np.array([[0], [0]])).get_inner_box(), color='g')
ztp.plot_zonotope(F_linear_multistep.W.get_bounding_box(), color='k')
# ztp.plot_zonotope(ztp.Zonotope(F_linear.A, np.array([[0], [0]])), color='b')
plt.savefig('destination_path.eps', format='eps')
plt.show()
a = 2

