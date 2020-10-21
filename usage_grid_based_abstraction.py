import numpy as np
from dynamics_library import SampleAndHold, PendulumCartContinuous
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
import zonotope_lib as ztp
import pwa_lib as pwa
import matplotlib.animation as animation

state_regions = []
state_regions_feedback = []
state_regions_alpha = []
x_len = 1.0
y_len = 10.0
region_shape = np.array([[-x_len / 2, x_len / 2], [-y_len / 2, y_len / 2]])
input_region = ztp.Box(np.array([[-0.5, 0.5]]))
x_start = -3.0
x_end = 3.0
y_start = -4.0
y_end = 4.0
n_time_steps = 10
unit_cell_size = np.array([[-.1, .1], [-.1, .1]])
unit_cell = ztp.Box(unit_cell_size)
for i in np.arange(x_start, x_end, x_len / 2):
    region_limits = copy(region_shape)
    region_limits[0, :] += i
    state_regions.append(ztp.Box(copy(region_limits)))
    state_regions_feedback.append(0)
    state_regions_alpha.append(0)
    ztp.plot_zonotope(state_regions[-1], fill=False)




f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)

F_linear = []
F_linear_multistep = []
feed_back_law = []
for id, reg in enumerate(state_regions):
    F_linear.append(pwa.get_affine_dynamics(F=F, state_region=reg, input_region=input_region))
    F_linear_multistep.append(pwa.get_multistep_system(affine_system=F_linear[-1], n_time_steps=n_time_steps))
    feed_back_law.append(pwa.synthesize_controller(F_linear_multistep[-1], unit_cell, input_region, 0.99 * unit_cell))


cells = []
winning_cells = []
for i in np.arange(x_start, x_end, unit_cell_size[0][1]*2):
    for j in np.arange(y_start, x_end, unit_cell_size[0][1] * 2):
        center = np.array([[i], [j]])
        new_cell = ztp.Box(center + unit_cell_size)
        cells.append(copy(new_cell))
        if np.all(np.abs(center) < 0.01) and np.all(np.abs(center) > -0.01):
            winning_cells.append(copy(new_cell))

solved_winning_cells = []

for i in range(1, 2):
    target = winning_cells.pop(0)
    for cid, c in reversed(list(enumerate(cells))):
        if pwa.can_reach(F_linear_multistep[6], c, target):
            winning_cells.append(copy(c))
            cells.pop(cid)
            ztp.plot_zonotope(c, color='g')
    solved_winning_cells.append(copy(target))
    plt.show()


