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

# Set industrial payoff parameters
a = 1000  # profit cap
#b1 = 0.05 # profit vs. water parameter
#b2 = 0.05 # profit vs. labor parameter
b1 = 1
b2 = 1
q = -0.5
c = 0.05  # extraction cost parameter
d = 10

# Set domestic user parameters
k = 2 # water access parameter (how steeply benefit of water rises)
p = 3 # out-of-system wellbeing/payoff
h = 0.06  # rate at which wage declines with excess labor
g = 0.01 # rate at which wage increases with marginal benefit of more labor
m = 0.8 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
W_min = 0

n = 20
dt = 0.08


num_points = 10

fee = 0
fee_cap = 0
fine = 0
fine_cap = 0
np.random.seed(1)
initial_points = doe.lhs(3, samples = num_points)
# Scale points ([R, U, W])
initial_points[:,0] = initial_points[:,0] * 100
initial_points[:,1] = initial_points[:,1] * 45
initial_points[:,2] = initial_points[:,2] * 20

# initialize sustinable equilibrium count
sust_eq = 0

plt.figure()
ax = plt.axes(projection='3d')

for i, point in enumerate(initial_points):
  R_0 = point[0]
  U_0 = point[1]
  W_0 = point[2]
  pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                  W_min, dt, R_0, W_0, U_0, fine_cap, fine)
  R, E, U, S, W, P, L, converged = pp.run()
  eff_W = np.zeros(len(R))
  R = np.array(R)/100
  # Plot effective wage instead
  eff_W = np.array(W[1:])*(np.array(L)/np.array(U[1:]))
  if R[-1] < 0.9:
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
ax.set_zlabel('Effective Wage ' + r'($\frac{L}{U}$W)')
ax.set_zlim(0,16)
ax.set_ylim(0,40)

