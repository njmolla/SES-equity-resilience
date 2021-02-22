import numpy as np
import matplotlib.pyplot as plt
import csv
import pyDOE as doe
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../')
from simulate import simulate_SES


def plot_equilibrium_basins(r, R_max, a, b1, b2, q, c, d, fine_cap, fine, fee_cap,
                            fee, k, p, h, g, m, W_min, dt, policy, num_points):

  """
  Produce scatterplot of randomly sampled initial conditions color coded by the
  equilibrium that they end up at. Saves a file with the equilibrium (whether it
  is a sustainable or collapse outcome) for each point.

  """

  # generate initial points using latin hypercube sampling

  np.random.seed(0)
  initial_points = doe.lhs(3, samples = num_points)
  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20


  sust_eq = 0
  # initialize matrix recording whether initial point leads to good or bad eq
  eq_condition = np.zeros(len(initial_points))
  # matrix for recording sustainable eq
  eq = []
  ax = plt.axes(projection='3d')

  for i, point in enumerate(initial_points):
    R_0 = point[0]
    U_0 = point[1]
    W_0 = point[2]
    pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                      W_min, dt, R_0, W_0, U_0, fine_cap, fine, fee_cap, fee)
    R, E, U, S, W, P, L, convergence = pp.run()

    if R[-1] > 90 and U[-1] < 1:
      eq_condition[i] = 0
      ax.scatter(R[0], U[0], W[0], s = (30,), marker = 'o', color = 'red', alpha = 0.3)
    else:
      ax.scatter(R[0], U[0], W[0], s = (30,), marker = 'o', color = 'blue', alpha = 0.3)
      ax.scatter(R[-1], U[-1], W[-1], s = (60,), marker = '*', color = 'b')
      eq_condition[i] = 1
      sust_eq += 1
      eq.append((R[-1],U[-1],W[-1]))


  ax.scatter(100, 0, 0, s = (60,), marker = 'X', color = 'red')
  ax.set_xlabel('Resource (R)')
  ax.set_ylabel('Population (U)')
  ax.set_zlabel('Wage (W)')
  if policy == 'fine':
    filename = 'scatterplot_data\\fine_eq_basins_c%s_%s_%s.csv'%(c, fine_cap, int(fine))
  else:
    filename = 'scatterplot_data\\fee_eq_basins_c2_%s_%s.csv'%(fee_cap, int(fee))
  with open(filename, 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(np.transpose([initial_points[:,0], initial_points[:,1], initial_points[:,2], eq_condition]))
    csvwriter.writerows(eq)
  sust_eq /= len(initial_points)

  return sust_eq, eq_condition, eq

#set resource parameters
r = 10
R_max = 100  # aquifer capacity

# Set industrial payoff parameters
a = 1000  # profit cap

b1 = 1
b2 = 1
q = -0.5
c = 0.07  # extraction cost parameter
d = 10

# Set domestic user parameters
k = 2 # water access parameter (how steeply benefit of water rises)
p = 3 # out-of-system wellbeing/payoff
h = 0.06 # rate at which wage declines with excess labor
g = 0.01 # rate at which wage increases with marginal benefit of more labor
m = 0.8 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
W_min = 0


dt = 0.08

# set policy
fine_cap = 3
fine = 200
fee_cap = 100
fee = 0
policy = 'fine'
num_points = 100

#sust_eq, eq_condition, eq = plot_equilibrium_basins(r, R_max, a, b1, b2, q, c, d, fine_cap, fine, fee_cap,
#                                  fee, k, p, h, g, m, W_min, dt, policy, num_points)

def plot_basin(file, num_points):
  data = np.loadtxt(file, delimiter = ',', max_rows = num_points)
  R_0 = data[:num_points,0]
  U_0 = data[:num_points,1]
  W_0 = data[:num_points,2]
  eq_condition = data[:num_points,3]
#  eq = np.loadtxt(file, delimiter = ',', skiprows = num_points)

  plt.figure()
  ax = plt.axes(projection='3d')
  #ax.scatter(eq[:,0], eq[:,1], eq[:,2], s = (60,), marker = '*', color = 'b')
  for i in range(len(R_0)):
    if eq_condition[i] == 0:
      ax.scatter(R_0[i], U_0[i], W_0[i], s = (30,), marker = 'o', color = 'red', alpha = 0.3)
    else:
      ax.scatter(R_0[i], U_0[i], W_0[i], s = (30,), marker = 'o', color = 'blue', alpha = 0.3)


#file = 'scatterplot_data\\fine_eq_basins_100_0.csv'
#plot_basin(file,num_points)
#

def basin_comparison(file1, file2, num_points):
  """
  Takes in two files with the equilibrium conditions and produces scatterplot
  showing how the basin of attraction of file2 differs from that of file1 (file1
  should be the baseline)
  """
  data_1 = np.loadtxt(file1, delimiter = ',', max_rows = num_points)
  data_2 = np.loadtxt(file2, delimiter = ',', max_rows = num_points)

  R_0 = data_1[:,0]
  U_0 = data_1[:,1]
  W_0 = data_1[:,2]
  eq_1 = data_1[:,3]
  eq_2 = data_2[:,3]

  both_fail = []
  fail_succeeds = []
  succeeds_fails = []
  both_succeed = []

  for i in range(len(R_0)):
    if eq_1[i] == 0:
      # both fail -> color red
      if eq_2[i] == 0:
        both_fail.append([R_0[i], U_0[i], W_0[i]])
      # baseline (policy 1) fails while policy 2 does not -> color green
      else:
        fail_succeeds.append([R_0[i], U_0[i], W_0[i]])
    else:
      # baseline succeeds, but policy 2 fails -> color orange
      if eq_2[i] == 0:
        succeeds_fails.append([R_0[i], U_0[i], W_0[i]])
      # both policies succeed (safe zone) -> color blue
      else:
        both_succeed.append([R_0[i], U_0[i], W_0[i]])
  return np.array(both_fail), np.array(fail_succeeds), np.array(succeeds_fails), np.array(both_succeed)




file1 = 'scatterplot_data\\fine_eq_basins_100_0.csv'
file2 = 'scatterplot_data\\fine_eq_basins_5_130.csv'
both_fail, fail_succeeds, succeeds_fails, both_succeed = basin_comparison(file1, file2, 200)


plt.figure()
ax = plt.axes(projection='3d')
if len(both_fail>0):
  ax.scatter(both_fail[:,0], both_fail[:,1], both_fail[:,2], s = (30,), marker = 'o', color = 'red', alpha = 0.3, label = 'Leads to collapse with and without policy')
if len(fail_succeeds>0):
  ax.scatter(fail_succeeds[:,0], fail_succeeds[:,1], fail_succeeds[:,2], s = (30,), marker = 'o', color = 'green', alpha = 0.3, label = 'Leads to collapse without policy, but sustainability with policy')
if len(succeeds_fails>0):
  ax.scatter(succeeds_fails[:,0], succeeds_fails[:,1], succeeds_fails[:,2], s = (30,), marker = 'o', color = 'orange', alpha = 0.3, label = 'Leads to collapse with policy, but sustainability without')
if len(both_succeed>0):
  ax.scatter(both_succeed[:,0], both_succeed[:,1], both_succeed[:,2], s = (30,), marker = 'o', color = 'blue', alpha = 0.3, label = 'Leads to sustainability with and without policy')
ax.set_xlabel('Resource (R)')
ax.set_ylabel('Population (U)')
ax.set_zlabel('Wage (W)')
ax.legend(loc = 'upper right')
plt.tight_layout()
#
##
