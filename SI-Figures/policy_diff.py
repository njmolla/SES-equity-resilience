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

num_processors = 3
#if comm.size != num_processors:
#  print('ERROR running on %d processors' % comm.size)
#  sys.exit()

def policy_diff(r = 10, R_max = 100, a = 1000, b1 = 1, b2 = 1, q = -0.5, c = 0.05,
                d = 100, k = 2, p = 3, h = 0.06, g = 0.01, m = 0.3,
                W_min = 0, dt = 0.08):
  '''
  Loop through several different caps to see if policies have any effect. Takes in
  usual parameters, returns whether or not any of the policies led to a different
  outcome
  '''
  num_points = 50
  np.random.seed(0)
  initial_points = doe.lhs(3, samples = num_points)
  # Scale points ([R, U, W])
  R_0 = initial_points[:,0] * 100
  U_0 = initial_points[:,1] * 45
  W_0 = initial_points[:,2] * 20

  amount = 500
  caps = np.linspace(1.5, 31.5, 11)
  caps = np.append(caps, 100)
  res = np.zeros(len(caps))

  for i, cap in enumerate(caps):
    for j in range(num_points):
      ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                              W_min, dt, R_0[j], W_0[j], U_0[j], cap, amount)
      R, E, U, S, W, P, L, convergence = ts.run()

      if R[-1] < 90 and U[-1] > 1:
        res[i] += 1

  return res/num_points

# --------------------------------------------------------------------------
# for running in parallel
# --------------------------------------------------------------------------
params = {}
params['c'] = np.linspace(0,2,45)
params['d'] = np.linspace(40,225,45)
params['p'] = np.linspace(0,35,40)
params['h'] = np.linspace(0,120,45)
params['g'] = np.linspace(0,0.1,45)
params['m'] = np.linspace(0,15,45)
params['r'] = np.linspace(4,16,45)

keys = ['p','h','r']
key = keys[comm.rank]
outcomes = np.zeros([45, 12])
for i in range(len(params[key])):
 outcomes[i] = policy_diff(**{key : params[key][i]})
 with open('%s_policydiff_2.csv'%(key), 'w+') as f:
   csvwriter = csv.writer(f)
   csvwriter.writerows(outcomes)


# --------------------------------------------------------------------------
# plot outcome of experiments
# --------------------------------------------------------------------------
param = 'p'
data = np.loadtxt('Figure_Code\\Bifurcation_Diagrams\\%s_policydiff.csv'%(param), delimiter = ',')
# get parameter values for which policy makes a difference
baseline_vals = data[:,5].reshape((40,1))
baseline_vals = np.broadcast_to(baseline_vals,np.shape(data[:,:-1]))
res_diff = data[:,:-1] - baseline_vals
diff_rows = np.any(res_diff[:,:-1]>0.02001,axis = 1)

baseline_vals_2 = data[:,9].reshape((45,1))
baseline_vals_2 = np.broadcast_to(baseline_vals,np.shape(data[:,:-1]))
res_diff_2 = data[:,:-1] - baseline_vals_2

fig, axarr = plt.subplots(2, 1, figsize=(8,8), sharex = 'col')
fig1 = axarr[0].imshow(res_diff, origin = 'lower', extent = [1.5,31.5,0,0.4], aspect = 30/(0.4))
fig2 = axarr[1].imshow(res_diff_2, origin = 'lower', extent = [1.5,31.5,2,10], aspect = 30/(10-2))
fig.colorbar(fig1,ax=axarr[0])
fig.colorbar(fig1,ax=axarr[1])

axarr[1].set_xlabel('Cap')
axarr[0].set_ylabel('%s'%(param))
axarr[1].set_ylabel('%s'%(param))
diff_params = params[param][diff_rows]
print(min(diff_params))
print(max(diff_params))
print(len(diff_params))
plt.figure()
cap_lb = 1.5
cap_ub = 31.5
p_lb = params[param][0]
p_ub = params[param][-1]
plt.imshow(res_diff, origin = 'lower', extent = [0,cap_ub,p_lb,p_ub], aspect = cap_ub/(p_ub-p_lb))
plt.colorbar()
plt.xlabel('Cap', fontsize = 'x-large')
plt.ylabel('%s'%(param), fontsize = 'x-large')