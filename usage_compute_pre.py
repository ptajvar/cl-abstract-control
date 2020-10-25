import numpy as np
from dynamics_library import SampleAndHold, PendulumCartContinuous
from copy import copy, deepcopy
import matplotlib
import matplotlib.pyplot as plt
import zonotope_lib as ztp
import pwa_lib as pwa
import matplotlib.animation as animation
import time
import pickle

f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)

input_min = -5
input_max = -input_min
input_box = ztp.Box(np.array([[input_min, input_max]]))
theta_min = -2.0
theta_max = 2.0
theta_dot_min = -3.2
theta_dot_max = 3.2
n_steps = 5
state_box = ztp.Box(np.array([[theta_min, theta_max], [theta_dot_min, theta_dot_max]]))

core_target = ztp.size_to_box(np.array([[0.1], [0.02]]))
# target = core_target
target, linear_ctrl = pwa.get_attraction(F, core_target, 0.1*input_box, n_steps)

H = pwa.PiecewiseAffineSys(state_box=state_box, input_box=input_box, refinement_list=[0])  # only refine angle
precision = 0.9*target

start_time = time.time()
H.compute_hybridization(F, precision, input_box=input_box, n_time_steps=n_steps)

with open('hybridization_5step_full.pickle', 'wb') as handle:
    pickle.dump(H, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("hybridization time: {}".format(time.time() - start_time))
print("Number of hybridization nodes: {}".format(H.size))
start_time = time.time()
iterations = 4
# cells = H.state_cell.get_bare_copy()
cells = pwa.StateCell(state_box.get_range())
target_list = [target]
colors = ['g', 'y', 'k', 'r', 'b']
tot_time = time.time()
ztp.plot_zonotope(target)
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
            if cell.stage is None and cell.layer < best_layer+2:
                best_layer = min(best_layer, cell.layer)
                # print(best_layer)
                cell.stage = i
                target_list.append(cell.as_box())
                # print("yoohoo")
        elif cell.children:
            cell_list += cell.children
print("total time {}".format(time.time() - tot_time))

for i in reversed(range(iterations)):
    cell_list = [cells]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.stage == i:
            ztp.plot_zonotope(cell.as_box(), color=colors[i % len(colors)], fill=False)
        else:
            cell_list += cell.children

with open('cell_control_5step_full.pickle', 'wb') as handle:
    pickle.dump(cells, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.show()
