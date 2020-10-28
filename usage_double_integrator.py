import pwa_lib as pwa
import zonotope_lib as ztp
import numpy as np
import time
from dynamics_library import DoubleIntegrator, SampleAndHold
import matplotlib.pyplot as plt
from copy import copy

f = DoubleIntegrator()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)

input_min = -1
input_max = -input_min
input_box = ztp.Box(np.array([[input_min, input_max]]))
x_min = -0.05
x_max = 0.05
v_min = -0.15
v_max = 0.15
n_steps = 10
state_box = ztp.Box(np.array([[x_min, x_max], [v_min, v_max]]))

target = ztp.size_to_box(np.array([[0.1], [0.1]]))
H = pwa.PiecewiseAffineSys(state_box=state_box, input_box=input_box, refinement_list=[0])  # only refine angle
precision = 0.9*target
H.compute_hybridization(F, precision, input_box=input_box, n_time_steps=n_steps)
print(H.size)

iterations = 1
# cells = H.state_cell.get_bare_copy()
cells = pwa.StateCell(state_box.get_range())
target_list = [target]
colors = ['g', 'y', 'k', 'r', 'b']
tot_time = time.time()
start_time = time.time()
for i in range(iterations):
    best_layer = 100
    while target_list:
        print("going after targets")
        tgt = target_list.pop(0)
        cells = pwa.compute_pre(H, cells, input_box, tgt)
        print("synthesis time: {}".format(time.time() - start_time))
        start_time = time.time()
    print("done with targets")
    cell_list = [cells]
    while cell_list:
        cell = cell_list.pop(0)
        assert isinstance(cell, pwa.StateCell)
        if cell.fully_solved():
            if cell.stage is None:
                cell.stage = i
            if cell.layer < best_layer+2:
                best_layer = min(best_layer, cell.layer)
                target_list.append(cell.as_box())
        elif cell.children:
            cell_list += cell.children
print("total time {}".format(time.time() - tot_time))

for i in reversed(range(iterations)):
    cell_list = [cells]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.is_winning:
            cell.stage = 0
            ztp.plot_zonotope(cell.as_box(), color=colors[i % len(colors)], fill=False)
        else:
            cell_list += cell.children

ztp.plot_zonotope(target, color='c')


x_0 = np.array([[-0.04], [-0.10]])
x = copy(x_0)
x_hist = [copy(x)]
u_hist = []
t_step = 0.01
control = np.array([[]])
for t in np.arange(0, 10, t_step):
    if control.size == 0:
        cell = cells.get_cell_min_stage(x)
        a = ztp.plot_zonotope(cell.as_box(), color='r', fill=False)
        plt.show()
        p = a.pop(-1)
        p.remove()
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




