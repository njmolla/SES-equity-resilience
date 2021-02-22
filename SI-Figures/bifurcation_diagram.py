import numpy as np
import matplotlib.pyplot as plt
import pyDOE as doe
from mpi4py import MPI
import sys
sys.path.append('../')
from simulate import simulate_SES
import csv
comm = MPI.COMM_WORLD
import sys

num_processors = 7
if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

def bifurcation(r = 10, R_max = 100, a = 1000, b1 = 1, b2 = 1, q = -0.5, c = 0.05,
                d = 100, k = 2, p = 3, h = 0.06, g = 0.01, m = 0.3,
                W_min = 0, dt = 0.08):
  '''
  Loop through several different caps to see if policies have any effect. Takes in
  usual parameters, returns whether or not any of the policies led to a different
  outcome
  '''
  num_points = 10
  np.random.seed(0)
  initial_points = doe.lhs(3, samples = num_points)
  # Scale points ([R, U, W])
  R_0 = initial_points[:,0] * 100
  U_0 = initial_points[:,1] * 45
  W_0 = initial_points[:,2] * 20

  eq_R = np.zeros([num_points])
  eq_U = np.zeros([num_points])
  eq_W = np.zeros([num_points])

  for i in range(num_points):
    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                            W_min, dt, R_0[i], U_0[i], W_0[i])
    R, E, U, S, W, P, L, convergence = ts.run()

    eq_R[i] = R[-1]
    eq_U[i] = U[-1]
    eq_W[i] = W[-1]

  return eq_R, eq_U, eq_W

# --------------------------------------------------------------------------
# For running in parallel
# --------------------------------------------------------------------------
params = {}
params['c'] = np.linspace(0,5,45)
params['d'] = np.linspace(0,250,45)
params['p'] = np.linspace(0,50,45)
params['h'] = np.linspace(0,120,45)
params['g'] = np.linspace(0,0.15,45)
params['m'] = np.linspace(0,15,45)
params['r'] = np.linspace(0,30,45)

keys = ['c','d','p','h','g','m','r']
key = keys[comm.rank]
eq_R = np.zeros([10, 45])
eq_U = np.zeros([10, 45])
eq_W = np.zeros([10, 45])

for i in range(len(params[key])):
  eq_R[:,i], eq_U[:,i], eq_W[:,i] = bifurcation(**{key : params[key][i]})
  with open('%s_bfd.csv'%(key), 'w+') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(eq_R.flatten())
    csvwriter.writerow(eq_U.flatten())
    csvwriter.writerow(eq_W.flatten())

# --------------------------------------------------------------------------
# Plotting (comment out for running on cluster)
# --------------------------------------------------------------------------
plt.figure()
param = 'r'
data = np.loadtxt('Bifurcation_Diagrams\\%s_bfd.csv'%(param), delimiter = ',')
param_range = list(params[param])*10
plt.plot(param_range,data[1],'.')
plt.xlabel('%s'%(param), fontsize = 'x-large')
plt.ylabel('Equilibrium Population', fontsize = 'x-large')