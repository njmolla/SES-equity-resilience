import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyDOE as doe

import sys
sys.path.append('../')
from simulate import simulate_SES
# Plot trajectories for several different initial conditions

# Plot timeseries for a single run
# set resource parameters
r = 10
R_max = 100  # aquifer capacity

dollar_scale = 0.1
# Set industrial payoff parameters
a = 10000 * dollar_scale  # profit cap
#b1 = 0.05 # profit vs. water parameter
#b2 = 0.05 # profit vs. labor parameter
b1 = 1
b2 = 1
q = -0.5
c = 0.5 * dollar_scale  # extraction cost parameter
d = 10

# Set domestic user parameters
k = 2 # water access parameter (how steeply benefit of water rises)
p = 30 * dollar_scale # out-of-system wellbeing/payoff
h = 0.6 * dollar_scale # rate at which wage declines with excess labor
g = 0.01 # rate at which wage increases with marginal benefit of more labor
m = 0.08 / dollar_scale # responsiveness of population to wellbeing (h*p > 1 with dt=1)
W_min = 0

r_mean = 10
r_stdev = 2
U_stdev = 2

# length of simulation
T = 50
n = 550

dt = 0.08


num_points = 30

fee = 0 * dollar_scale
fee_cap = 0
np.random.seed(2)
initial_points = doe.lhs(3, samples = num_points)
# Scale points ([R, U, W])
initial_points[:,0] = initial_points[:,0] * 100
initial_points[:,1] = initial_points[:,1] * 45
initial_points[:,2] = initial_points[:,2] * 200 * dollar_scale

# initialize sustinable equilibrium count
sust_eq = 0

plt.figure()
ax = plt.axes(projection='3d')

for i, point in enumerate(initial_points):
  R_0 = point[0]
  U_0 = point[1]
  W_0 = point[2]
  pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                  W_min, dt, R_0, W_0, U_0)
  R, E, U, S, W, P, L = pp.run()
  eff_W = np.zeros(len(R))
  # Plot effective wage instead
  eff_W = np.array(W[1:])*(np.array(L)/np.array(U[1:]))
  if R[-1] < 90:
    ax.plot3D(R[1:], U[1:], eff_W, color = 'c', linewidth = 0.75)
    ax.scatter(R[1], U[1], eff_W[0], s = (10,), marker = 'o', color = 'c')
    ax.scatter(R[-1], U[-1], eff_W[-1], s = (35,), marker = '*', color = 'b')
    sust_eq += 1
  else:
    ax.plot3D(R[1:], U[1:], eff_W, 'k', linewidth = 0.75)
    ax.scatter(R[1], U[1], eff_W[0], s = (10,), marker = 'o', color = 'black')
    ax.scatter(R[-1], U[-1], eff_W[-1], s = (35,), marker = 'X', color = 'red')
  ax.set_xlabel('Resource (R)')
  ax.set_ylabel('Population (U)')
  ax.set_zlabel('Wage (W)')

ax.view_init(30, -60)
print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))
