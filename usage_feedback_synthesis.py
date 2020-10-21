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
state_region = ztp.Box(np.array([[-1, 1], [-1, 1]]))
input_region = ztp.Box(np.array([[-0.5, 0.5]]))
F_linear = pwa.get_affine_dynamics(F=F, state_region=state_region, input_region=input_region, n_sample=200)
n_time_steps = 20
F_linear_multistep = pwa.get_multistep_system(affine_system=F_linear, n_time_steps=n_time_steps)
feed_back_law = pwa.synthesize_controller(F_linear_multistep, 0.2*state_region, input_region, 10*state_region)
print(feed_back_law)

cell_0 = ztp.Box(np.array([[-0.1, 0.1], [-0.1, 0.1]]))


r_set = copy(cell_0)
cell_hist = []
for k, t in enumerate(np.arange(0, run_time, sample_time)):
    row = k % feed_back_law.shape[0]
    ztp.plot_zonotope(r_set, fill=True)
    ztp.plot_zonotope(r_set, fill=False, color='k-')
    cell_hist.append(copy(r_set))
    F_linear_cl = F_linear.get_closed_loop(feed_back_law[row:row+1, :])
    r_set = F_linear_cl.compute_reachable_set(copy(r_set))
plt.show()
