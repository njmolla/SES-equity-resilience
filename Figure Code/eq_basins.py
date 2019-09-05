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
  initial_points[:,2] = initial_points[:,2] * 200 * dollar_scale

  plt.figure()
  ax = plt.axes(projection='3d')

  sust_eq = 0
  # initialize matrix recording whether initial point leads to good or bad eq
  eq_condition = np.zeros(len(initial_points))

  for i, point in enumerate(initial_points):
    R_0 = point[0]
    U_0 = point[1]
    W_0 = point[2]
    pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                      W_min, dt, R_0, W_0, U_0, fine_cap, fine, fee_cap, fee)
    R, E, U, S, W, P, L = pp.run()

    if R[-1] > 90 and U[-1] < 1:
      ax.scatter(R[0], U[0], W[0], s = (30,), marker = 'o', color = 'red', alpha = 0.3)
      eq_condition[i] = 0
    else:
      ax.scatter(R[0], U[0], W[0], s = (30,), marker = 'o', color = 'blue', alpha = 0.3)
      ax.scatter(R[-1], U[-1], W[-1], s = (60,), marker = '*', color = 'b')
      eq_condition[i] = 1
      sust_eq += 1

  ax.scatter(100, 0, 0, s = (60,), marker = 'X', color = 'red')
  ax.set_xlabel('Resource (R)')
  ax.set_ylabel('Population (U)')
  ax.set_zlabel('Wage (W)')
  if policy == 'fine':
    filename = 'fine_eq_basins_%s_%s.csv'%(fine_cap, int(fine))
  else:
    filename = 'fee_eq_basins_%s_%s.csv'%(fee_cap, int(fee))
  with open(filename, 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(np.transpose([initial_points[:,0], initial_points[:,1], initial_points[:,2], eq_condition]))
  sust_eq /= len(initial_points)

  return sust_eq, eq_condition

# set resource parameters
r = 10
R_max = 100  # aquifer capacity

dollar_scale = 0.1
# Set industrial payoff parameters
a = 10000 * dollar_scale  # profit cap
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

# set policy
fine_cap = 0
fine = 0
fee_cap = 5
fee = 30 * dollar_scale # scale down by 10
policy = 'fee'
num_points = 1000

sust_eq, eq_condition = plot_equilibrium_basins(r, R_max, a, b1, b2, q, c, d, fine_cap, fine, fee_cap,
                                  fee, k, p, h, g, m, W_min, dt, policy, num_points)


def basin_comparison(file1, file2):
  """
  Takes in two files with the equilibrium conditions and produces scatterplot
  showing how the basin of attraction of file2 differs from that of file1 (file1)
  should be the baseline)
  """
  data_1 = np.loadtxt(file_1, delimiter = ',')
  data_2 = np.loadtxt(file_2, delimiter = ',')

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

file_1 = 'fine_eq_basins_200_0.csv'
file_2 = 'fine_eq_basins_5_125.csv'
#both_fail, fail_succeeds, succeeds_fails, both_succeed = basin_comparison(file_1, file_2)
#
#plt.figure()
#ax = plt.axes(projection='3d')
#if len(both_fail>0):
#  ax.scatter(both_fail[:,0], both_fail[:,1], both_fail[:,2], s = (30,), marker = 'o', color = 'red', alpha = 0.3, label = 'Leads to collapse with and without policy')
#if len(fail_succeeds>0):
#  ax.scatter(fail_succeeds[:,0], fail_succeeds[:,1], fail_succeeds[:,2], s = (30,), marker = 'o', color = 'green', alpha = 0.3, label = 'Leads to collapse without policy, but sustainability with policy')
#if len(succeeds_fails>0):
#  ax.scatter(succeeds_fails[:,0], succeeds_fails[:,1], succeeds_fails[:,2], s = (30,), marker = 'o', color = 'orange', alpha = 0.3, label = 'Leads to collapse with policy, but sustainability without')
#if len(both_succeed>0):
#  ax.scatter(both_succeed[:,0], both_succeed[:,1], both_succeed[:,2], s = (30,), marker = 'o', color = 'blue', alpha = 0.3, label = 'Leads to sustainability with and without policy')
#ax.set_xlabel('Resource (R)')
#ax.set_ylabel('Population (U)')
#ax.set_zlabel('Wage (W)')
#ax.legend(loc = 'upper right')
#plt.tight_layout()
#
#
