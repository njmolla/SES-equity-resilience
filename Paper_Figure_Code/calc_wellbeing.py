import numpy as np
import glob
from restitch_colormap import avg_trajectories, wellbeing, return_eq, return_snapshot
import pickle
import bz2
import sys
import csv
sys.path.append('../')
from simulate import simulate_SES
#import pyDOE as doe


def resource_from_extraction(R_0,trajectory):
  R = np.zeros(len(trajectory)+1)
  R[0] = R_0
  # set resource parameters
  r = 10
  R_max = 100  # aquifer capacity

  # Set industrial payoff parameters
  a = 1000  # profit cap
  b1 = 1
  b2 = 1
  q = -0.5
  c = 0.05  # extraction cost parameter
  d = 100

  # Set domestic user parameters
  k = 2 # water access parameter (how steeply benefit of water rises)
  p = 3 # out-of-system wellbeing/payoff
  h = 0.06 # rate at which wage declines with excess labor
  g = 0.01 # rate at which wage increases with marginal benefit of more labor
  m = 0.8 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
  W_min = 0
  dt = 0.08
  W_0 = 0
  U_0 = 0
  for i,E in enumerate(trajectory):
    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                            W_min, dt, R_0, W_0, U_0)
    R[i+1] = ts.resource(R[i], E)
  return R

def _wage_key_func(x):
  """Returns the policy number portion of filename."""
  return int(x[len('wage_') : -2])

def _r_key_func(x):
  """Returns the policy number portion of filename."""
  return int(x[len('resource_') : -2])

def _pop_key_func(x):
  """Returns the policy number portion of filename."""
  return int(x[len('pop_') : -2])

def _labor_key_func(x):
  """Returns the policy number portion of filename."""
  return int(x[len('labor_') : -2])

def calculate_wellbeing(transient_data = False):
  """
  transient_data is a boolean. if true, calculate wellbeing avg over different time lengths. otherwise,
  calculate just the equilibrium colormap
  """
  num_points = 40
  num_trajectories = 80
  resource_files = sorted(glob.glob('resource'+'_*.p'), key = _r_key_func)
  wage_files = sorted(glob.glob('wage'+'_*.p'), key = _wage_key_func)
  pop_files = sorted(glob.glob('pop'+'_*.p'), key = _pop_key_func)

#  # Get R initial points
#  np.random.seed(1)
#  num_trajectories = 100
#  initial_points = doe.lhs(3, samples = num_trajectories)
#  # Scale points ([R, U, W])
#  initial_R = initial_points[:,0] * 100

  if transient_data:
    t_lengths = np.arange(10,2000,10)
    cmap_arr = np.zeros((len(t_lengths), num_points, num_points))

  else:
    cmap_arr = np.zeros((num_points, num_points))

  # import all of the files (just using resource files to get piece numbers)
  for k, file in enumerate(resource_files):
    n = int(file[len('resource' + '_'):len(file)-2]) # piece number
    y_index = int(n/num_points)
    x_index = int(n%num_points)
    # import data
    with bz2.BZ2File(file, 'rb') as f:
      resource_data = pickle.load(f)
    with bz2.BZ2File(wage_files[k], 'rb') as f:
      wage_data = pickle.load(f)
    with bz2.BZ2File(pop_files[k], 'rb') as f:
      pop_data = pickle.load(f)
    with bz2.BZ2File(pop_files[k], 'rb') as f:
      labor_data = pickle.load(f)

#    resource_data = [0]*num_trajectories
#    for i, trajectory in enumerate(extraction_data):
#      resource_data[i] = resource_from_extraction(initial_R[i],extraction_data[i])
    wellbeing_trajectories = wellbeing(resource_data, wage_data, pop_data, labor_data, num_trajectories)

    if transient_data:
      for j,t_length in enumerate(t_lengths):
        value = return_snapshot(wellbeing_trajectories, t_length)
        cmap_arr[j, x_index, y_index] = value
    else:
      eq = return_eq(wellbeing_trajectories)
      cmap_arr[x_index, y_index] = eq

  if transient_data:
    with open('wellbeing_snapshots.p', 'wb') as f:
      pickle.dump(cmap_arr, f)

  else:
    with open('fee_eq_wellbeing.csv', 'w+') as f:
      csvwriter = csv.writer(f)
      csvwriter.writerows(cmap_arr)


calculate_wellbeing(transient_data = True)


