import numpy as np
from dynamics_library import SampleAndHold, PendulumCartContinuous
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from zonotope_lib import Zonotope, Box
from pwa_lib import *
import matplotlib.animation as animation

f = PendulumCartContinuous()
sample_time = 0.01
F = SampleAndHold(continuous_function=f, sample_time=sample_time)
x_0 = np.array([1, 1])
run_time = 2
t_range = np.arange(0, run_time, sample_time)
x = deepcopy(x_0)
x_hist = np.zeros((2, len(t_range)))
for i, t in enumerate(t_range):
    x_hist[:, i] = x.reshape((2,))
    x = F(x, np.array([0]))

# finding invariant set around the upright position
target_point = np.array([0, 0]).reshape((2, 1))
tolerance = np.array([1, 1]).reshape((2, 1))
target_region = np.concatenate((target_point - tolerance, target_point + tolerance), axis=1)
target_set = Box(target_region)
input_set = Box(np.array([-10, 10]).reshape(1, 2))
affine_system = get_affine_dynamics(F, target_set, input_set)
n_time_steps = 10
multistep_system = get_multistep_system(affine_system=affine_system, n_time_steps=n_time_steps)
feed_back_law = synthesize_controller(multistep_system, 0.2*target_set, input_set, 0.05*target_set)

# running the closed loop system

x = deepcopy(x_0)
x_hist = np.zeros((2, len(t_range)))
feed_back_plan = np.zeros((2, 0))
for i, t in enumerate(t_range):
    if feed_back_plan.shape[1] == 0:
        feed_back_plan = np.matmul(feed_back_law, x)
        feed_back_plan = feed_back_plan.reshape((1, feed_back_plan.shape[0]))
    u = feed_back_plan[:, 0]
    if u[0] > 10.0:
        u[0] = 10.0
    if u[0] < -10.0:
        u[0] = -10.0
    x_hist[:, i] = x.reshape((2,))
    x = F(x, u)
    feed_back_plan = np.delete(feed_back_plan, 0, axis=1)

# plotting
fig = plt.figure(figsize=(10, 6))
plt.xlim(0, run_time)
plt.ylim(np.min(x_hist[0, :]), np.max(x_hist[0, :]))
n_frames = 50


def animate(i):
    ind = int(i * len(t_range) / n_frames)
    plt.plot(t_range[0:ind], x_hist[0, 0:ind], 'g')


Writer = animation.writers['pillow']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=n_frames, repeat=True)
plt.show()
# ani.save('pendulum.gif', writer=writer)

input()
