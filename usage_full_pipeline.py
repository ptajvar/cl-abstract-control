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

# Hybridization
input_min = -0.5
input_max = 0.5
input_region = ztp.Box(np.array([[input_min, input_max]]))
theta_min = -2.5
theta_max = 2.5
region_theta_range = 0.5
region_theta_step = region_theta_range / 2

theta_dot_min = -3.5
theta_dot_max = 3.5

state_regions = []
state_regions_F = []
for th in np.arange(theta_min, theta_max, region_theta_step):
    if th + region_theta_range > theta_max:
        break
    state_region = ztp.Box(np.array([[th, th + region_theta_range], [theta_dot_min, theta_dot_max]]))
    state_regions.append(deepcopy(state_region))
    F_linear = pwa.get_affine_dynamics(F=F, state_region=state_region, input_region=input_region, n_sample=1000)
    state_regions_F.append(deepcopy(F_linear))

target = ztp.size_to_box(np.array([[0.3], [1]]))

n_time_steps = 10
regions_F_multistep = []
regions_F_multistep_cl = []

min_cell_size = np.zeros((2, 1))
for F in state_regions_F:
    F_linear_multistep = pwa.get_multistep_system(affine_system=F, n_time_steps=n_time_steps)
    regions_F_multistep.append(deepcopy(F_linear_multistep))
    success, feedback_rule, alpha, F_multistep_cl = pwa.synthesize_controller(F_linear_multistep,
                                                                              state_cell=1.1 * target,
                                                                              input_range=input_region,
                                                                              target_size=target)
    if success:
        regions_F_multistep_cl.append(deepcopy(F_multistep_cl))
    else:
        regions_F_multistep_cl.append(deepcopy(F_linear_multistep))


    # input_reach = ztp.Zonotope(np.array([[], []]), np.array([[0], [0]]))
    # n_inputs = input_region.ndim
    # for i in range(n_time_steps):
    #     input_reach = input_reach + F_linear_multistep.B[:, n_inputs * i:n_inputs * (i + 1)] * input_region
    # ztp.plot_zonotope(input_reach, fill=False, color='g')
    # ztp.plot_zonotope(F_linear_multistep.W.get_inner_box(), fill=False, color='k')
    # plt.show()
    # print(input_reach.get_bounding_box_size())
    # print("-------------")
    min_cell_size = np.maximum(min_cell_size, F_linear_multistep.W.get_bounding_box_size())

# assert np.all(target.get_bounding_box_size() > min_cell_size)
# Control Synthesis
winning_set_resolved = []
winning_set_unresolved = []
partial_winning_set = []

cell_list = []
for i in np.arange(theta_min, theta_max, target.get_bounding_box_size()[0]):
    for j in np.arange(theta_dot_min, theta_dot_max, target.get_bounding_box_size()[1]):
        lb = np.array([[i], [j]])
        ub = lb + target.get_bounding_box_size()
        range = np.concatenate((lb, ub), axis=1)
        new_cell = pwa.StateCell(range)
        cell_list.append(deepcopy(new_cell))
        ztp.plot_zonotope(new_cell.as_box(), color='k', fill=False)

winning_set_unresolved.append(pwa.StateCell(target.get_range()))

while winning_set_unresolved:
    # if len(winning_set_unresolved) > 3:
    #     break
    temp_target = winning_set_unresolved.pop()
    winning_set_resolved.append(temp_target)
    temp_box = temp_target.as_box()
    temp_theta_range = temp_target.range[0, :]
    theta_in = 0
    best_region = -1
    for r_id, r in enumerate(np.arange(theta_min, theta_max, region_theta_step)):
        dist = np.minimum(temp_theta_range[0] - r, r + region_theta_range - temp_theta_range[1])
        if dist > theta_in:
            theta_in = dist
            best_region = r_id
    if best_region < 0 or best_region >= len(regions_F_multistep_cl):
        continue
    dynamics = regions_F_multistep_cl[best_region]

    for c_id, c in reversed(list(enumerate(cell_list))):
        reach_approx = dynamics.A * c.as_box() + dynamics.C + dynamics.W
        tolerance = temp_box.get_bounding_box_size() - dynamics.W.get_bounding_box_size() - (dynamics.A * c.as_box(

        )).get_bounding_box_size()
        if np.any(tolerance < 0):
            print("unstable dynamics")
            # cell_list.pop(c_id)
            continue
        # if not temp_box.intersects(reach_approx.get_bounding_box()):
        #     continue
        success, u = pwa.steer(affine_system=dynamics, start_state=c.as_box().center, target_state=temp_box.center,
                               input_range=input_region, tolerance=tolerance)
        if success:
            cell_list.pop(c_id)
            winning_set_unresolved.append(c)
            ztp.plot_zonotope(c.as_box(), color='g')
plt.show()

        # for u in np.linspace(input_min, input_max, 20):
        #     reachable = dynamics.A * c.as_box() + dynamics.B @ (u * np.ones((dynamics.B.shape[1], 1))) + dynamics.C + \
        #                 dynamics.W.get_bounding_box()
        #     ztp.plot_zonotope(reachable, color='g', fill=False)
        #     if c.as_box().contains(reachable.get_bounding_box()):
        #         success = True
        #         cell_list.pop(c_id)
        #         winning_set_unresolved.append(c)
        #         ztp.plot_zonotope(reachable, color='g', fill=False)
        #         print("yohoo")
        #         break


plt.show()
