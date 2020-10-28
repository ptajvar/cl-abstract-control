import pickle
import zonotope_lib as ztp
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import pwa_lib as pwa
from dynamics_library import *

with open('cell_control_5step_big', 'rb') as handle:
    cells_big = pickle.load(handle)
assert isinstance(cells_big, pwa.StateCell)
colors = ['g', 'y', 'k', 'r', 'b']
iterations = 4
for i in reversed(range(iterations)):
    cell_list = [cells_big]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.is_winning:
            print(cell.layer)
            print(cell.feedfwd_control)
            cell.stage = 0
            fig = ztp.plot_zonotope(cell.as_box(), color=colors[i % len(colors)], fill=True)
            fig = ztp.plot_zonotope(cell.as_box(), color='w', fill=False)
        else:
            cell_list += cell.children
# plt.show()
with open('cell_control_5step_fine', 'rb') as handle:
    cells_fine = pickle.load(handle)
assert isinstance(cells_fine, pwa.StateCell)
colors = ['g', 'y', 'k', 'r', 'b']
iterations = 4
for i in reversed(range(iterations)):
    cell_list = [cells_fine]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.is_winning and cell.stage == i:
            print(cell.layer)
            print(cell.feedfwd_control)
            fig = ztp.plot_zonotope(cell.as_box(), color=colors[i % len(colors)], fill=True)
            fig = ztp.plot_zonotope(cell.as_box(), color='w', fill=False)
        else:
            cell_list += cell.children
# plt.show()
plt.savefig('win_set.svg', format='svg', dpi=1200)

# running the sytem
f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)

x_0 = np.array([[0.3], [0.3]])
x = copy(x_0)
x_hist = [copy(x)]
u_hist = []
t_step = 0.01
control = np.array([[]])
for t in np.arange(0, 0.3, t_step):
    if control.size == 0:
        cell = cells_fine.get_cell_min_stage(x)
        if cell is None:
            cell = cells_big.get_cell_min_stage(x)
        if cell.feedback_control is None:
            control = cell.feedfwd_control
        else:
            control = cell.feedback_control @ (x - cell.as_box().center) + cell.feedfwd_control
    u = control[0].reshape((1, 1))
    control = np.delete(control, 0)
    plt.plot(x[0], x[1], 'b*')
    x = F(x, u)
    u_hist.append(copy(u))
    x_hist.append(copy(x))

plt.show()