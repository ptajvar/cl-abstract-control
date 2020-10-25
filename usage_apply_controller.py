import pickle
import zonotope_lib as ztp
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
with open('cell_control_5step_full.pickle', 'rb') as handle:
    cells = pickle.load(handle)

colors = ['g', 'y', 'k', 'r', 'b']
iterations = 4
for i in reversed(range(iterations)):
    cell_list = [cells]
    while cell_list:
        cell = cell_list.pop(0)
        if cell.stage == i:
            fig = ztp.plot_zonotope(cell.as_box(), color=colors[i % len(colors)], fill=True)
            fig = ztp.plot_zonotope(cell.as_box(), color='w', fill=False)
        else:
            cell_list += cell.children
# plt.show()
plt.savefig('win_set.svg', format='svg', dpi=1200)

x_0 = np.array([[0], [0]])
x = copy(x_0)
t_step = 0.01
control = []
for t in np.arange(0, 0.5, t_step):
    if not control:
        cells.get_cell_minstage(x)



