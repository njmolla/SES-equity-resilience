import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import bz2
import sys
sys.path.append('../')
from simulate import simulate_SES

def return_eq(data):
  """
  Extracts last point from each trajectory and averages them for each policy
  """
  eq = 0
  for i in range(len(data)):
    eq += data[i][-1]
  return eq/100

def return_snapshot(data, t):
  """
  Extracts t point from each trajectory and averages them for each policy
  """
  t_data = 0
  for i in range(len(data)):
    if t >= len(data[i]):
      t_data += data[i][-1]
    else:
      t_data += data[i][t]
  return t_data/len(data)

def wellbeing(resource_data, wage_data, pop_data, labor_data, num_points):
  """
  calculates wellbeing trajectories for each policy
  """
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
  m = 0.3 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
  W_min = 0
  dt = 0.08
  W_0 = 0
  U_0 = 0
  R_0 = 0

  wellbeing_trajectories = [0]*num_points
  ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                        W_min, dt, R_0, W_0, U_0)

  for i in range(num_points):
    water_access_data = np.array(ts.water_access(np.array(resource_data[i])))[1:]
    wage_data[i] = np.array(wage_data[i][1:])
    wellbeing_trajectories[i] = water_access_data*wage_data[i]*labor_data[i]
  return wellbeing_trajectories


def avg_trajectories(data, t_length):
  """
  Averages first t_length timesteps of each trajectory and averages them for each policy
  """
  policy_avg = 0
  num_trajectories = len(data)
  for i in range(num_trajectories):
    traj_avg = np.average(data[i][0:t_length])
    if t_length > len(data[i]):
      policy_avg += (traj_avg*len(data[i]) + data[i][-1]*(t_length - len(data[i])))/t_length
    else:
      policy_avg += traj_avg
  return policy_avg/num_trajectories

def calculate_resilience(data):
  """
  Counts number of trajectories that reach the sustainable equilibrium (using the population in
  this case)
  """
  num_sustainable = 0
  num_trajectories = len(data)
  for i in range(num_trajectories):
    if data[i][-1] > 0.1:
      num_sustainable += 1
  resilience = num_sustainable/num_trajectories
  return resilience


def restitch_data(folder, cmap_type, num_points, aggregation_method, *args):
  """
  Combines the colormap data that was broken up for parallelization back into
  one single array. folder is the location of all of the colormap data, and the
  cmap_type is the variable being plotted, of which
  there are four:
    payoff for industrial users ("payoff")
    domestic user population ("pop")
    wage
    extraction

  num_points is the number of points for the cap and amount (currently 50); aggregation
  method is the method for turning the trajectories for a given policy into a single value,
  either by averaging the final (equilibrium) value, averaging over a given time period,
  or counting the proportion of sustainable outcomes
  """
  cmap_arr = np.zeros((num_points, num_points))
  files = glob.glob(folder + cmap_type + '_*.p')
  for file in files:
    n = int(file[len(folder + cmap_type + '_'):len(file)-2]) # piece number
    y_index = int(n/num_points)
    x_index = int(n%num_points)
    with bz2.BZ2File(file, 'rb') as f:
      data = pickle.load(f)
    value = aggregation_method(data,*args)
    cmap_arr[x_index, y_index] = value
  return cmap_arr

def fix_colormap(cmap):
  cmap_fixed = np.zeros(np.shape(cmap))

  for i in range(64):
    x = int(5*(i%8))
    y = 5*(i//8)
    cmap_fixed[x:x + 5, y:y+5] = np.transpose(cmap[x:x + 5, y:y+5])
  return cmap_fixed


if __name__ == "__main__":
  policy = 'fee'


  with open("SI-Figures\\parameter_experiments\\p\\payoff_snapshots.p", 'rb') as f:
    payoff_data = pickle.load(f)
  payoff_data = payoff_data[1:] # accidentally got an extra point
  payoff_data = np.transpose(payoff_data, (0,2,1))

  with open("SI-Figures\\parameter_experiments\\p\\wellbeing_snapshots.p", 'rb') as f:
    wellbeing_data = pickle.load(f)
#
  t_lengths = np.arange(10,2000,10) * 0.04

#  for i in range(len(t_lengths)):
#    payoff_data[i] = fix_colormap(payoff_data[i])
#    wellbeing_data[i] = fix_colormap(wellbeing_data[i])
##
  cmap_dim = np.shape(wellbeing_data)[1]
  denom = np.zeros((len(wellbeing_data),1,1))
  denom[:,0,0] = np.amax(np.amax(payoff_data + wellbeing_data,axis = 1),axis=1)
  denom = np.broadcast_to(denom,np.shape(wellbeing_data))
  equity_data = np.where(payoff_data<0.1,0,wellbeing_data/denom)
  equity_data = np.transpose(equity_data, (0,2,1))
#
  resilience_map = np.loadtxt("SI-Figures\\parameter_experiments\\p\\resilience.csv", delimiter = ',')

  resilience_map = np.transpose(resilience_map)
  eq_payoff_map = payoff_data[-1]

  eq_equity_map = equity_data[-1]
  eq_equity_map = np.where(eq_payoff_map<0.001,0,eq_equity_map/(np.amax(eq_payoff_map + eq_equity_map)))
  eq_equity_map = np.transpose(eq_equity_map)

  # Plot fee colormaps

  # set extent of colormaps
  cap_upper_bound = 10 # upper bound of fine thresholds
  amount_upper_bound = 40 # upper bound of fine or fee amounts
  t1 = 11
  t2 = 30
  plot_max = np.amax(equity_data)
  fig, axarr = plt.subplots(1, 3, figsize=(8,8), sharex = 'col')
  axarr[0].set_title('Equity (190)', fontsize = 'x-large')
  equity_1 = axarr[0].imshow(equity_data[t1], origin = 'lower', vmin = 0,
              vmax = plot_max, extent = [0,cap_upper_bound,0,amount_upper_bound],
              aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
  axarr[0].set_ylabel('Fee Amount')

  axarr[1].set_title('Equity (420)', fontsize = 'x-large')
  equity_2 = axarr[1].imshow(equity_data[t2], origin = 'lower', vmin = 0,
              vmax = plot_max, extent =
                [0,cap_upper_bound,0,amount_upper_bound],
                aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')

  axarr[2].set_title('Equilibrium Equity', fontsize = 'x-large')
  eq_equity = axarr[2].imshow(eq_equity_map, origin = 'lower', vmin = 0,
                 vmax = 1, extent = [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
  axarr[0].set_xlabel('Fee Threshold',fontsize = 'x-large')
  axarr[1].set_xlabel('Fee Threshold',fontsize = 'x-large')
  axarr[2].set_xlabel('Fee Threshold',fontsize = 'x-large')

  tol = 0.95

  equity_1_max = equity_data[t1] >= tol * np.amax(equity_data[t1])
  payoff_1_max = payoff_data[t1] >= tol * np.amax(payoff_data[t1])
  equity_2_max = equity_data[t2] >= tol * np.amax(equity_data[t2])
  payoff_2_max = payoff_data[t2] >= tol * np.amax(payoff_data[t2])
  eq_equity_max = eq_equity_map >= tol * np.amax(eq_equity_map)
  eq_payoff_max = eq_payoff_map >= tol * np.amax(eq_payoff_map)
  resilience_max = resilience_map >= tol * np.amax(resilience_map)

  def mix_colors(cf, cb):
      a = cb[-1] + cf[-1] - cb[-1] * cf[-1] # fixed alpha calculation
      r = (cf[0] * cf[-1] + cb[0] * cb[-1] * (1 - cf[-1])) / a
      g = (cf[1] * cf[-1] + cb[1] * cb[-1] * (1 - cf[-1])) / a
      b = (cf[2] * cf[-1] + cb[2] * cb[-1] * (1 - cf[-1])) / a
      return [r,g,b,a]


  # Where we set the RGB for each pixel
  payoff_color = [0.40392157, 0, 0.05098039, 0.4] #reddish
  equity_color = [0.03137255, 0.18823529, 0.41960784, 0.6] #bluish
  overlap_color = mix_colors(equity_color, payoff_color)

  thres_1 = np.zeros((cmap_dim,cmap_dim,4))
  thres_1[equity_1_max] = equity_color
  thres_1[~equity_1_max] = [1,1,1,1]
  thres_1[np.logical_and(payoff_1_max, ~equity_1_max)] = payoff_color
  thres_1[np.logical_and(payoff_1_max, equity_1_max)] = overlap_color

  thres_2 = np.zeros((cmap_dim,cmap_dim,4))
  thres_2[equity_2_max] = equity_color
  thres_2[~equity_2_max] = [1,1,1,1]
  thres_2[np.logical_and(payoff_2_max, ~equity_2_max)] = payoff_color
  thres_2[np.logical_and(payoff_2_max, equity_2_max)] = overlap_color

  eq_thres = np.zeros((cmap_dim,cmap_dim,4))
  eq_thres[eq_equity_max] = equity_color
  eq_thres[~eq_equity_max] = [1,1,1,1]
  eq_thres[np.logical_and(eq_payoff_max, ~eq_equity_max)] = payoff_color
  eq_thres[np.logical_and(eq_payoff_max, eq_equity_max)] = overlap_color

  fig, axarr = plt.subplots(3, 1, figsize=(8,8), sharex = 'col')
  axarr[0].imshow(thres_1,interpolation='nearest',origin = 'lower', extent =
                   [0,cap_upper_bound,0,amount_upper_bound],
                   aspect = cap_upper_bound/amount_upper_bound)
  axarr[0].contour(resilience_max, [1], extent =
                   [0,cap_upper_bound,0,amount_upper_bound], colors = 'k')
  axarr[0].set_ylabel('Fee Amount', fontsize = 'x-large')

  axarr[1].imshow(thres_2,interpolation='nearest',origin = 'lower', extent =
                   [0,cap_upper_bound,0,amount_upper_bound],
                   aspect = cap_upper_bound/amount_upper_bound)
  axarr[1].contour(resilience_max, [1], extent =
                   [0,cap_upper_bound,0,amount_upper_bound], colors = 'k')
  axarr[1].set_ylabel('Fee Amount', fontsize = 'x-large')

  axarr[2].imshow(eq_thres,interpolation='nearest',origin = 'lower', extent =
                   [0,cap_upper_bound,0,amount_upper_bound],
                   aspect = cap_upper_bound/amount_upper_bound)
  axarr[2].contour(resilience_max, [1], extent =
                   [0,cap_upper_bound,0,amount_upper_bound], colors = 'k')
  axarr[2].set_ylabel('Fee Amount', fontsize = 'x-large')


  axarr[2].set_xlabel('Fee Threshold', fontsize = 'x-large')
#  axarr[1].set_xlabel('Fee Threshold', fontsize = 'x-large')
#  axarr[2].set_xlabel('Fee Threshold', fontsize = 'x-large')

  # create a patch (proxy artist) for every color
  patches = [mpatches.Patch(color = payoff_color, label='Profitable Outcomes', alpha = 0.6),
              mpatches.Patch(color = equity_color, label='Equitable outcomes', alpha = 0.6),
              mpatches.Patch(edgecolor = 'black', facecolor = 'white', linewidth = 1.5, label='High resilience')]
  plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), borderaxespad=0. )

  plt.tight_layout()


# --------------------------------------------------------------------------
# Plotting equilibrium colormaps
# --------------------------------------------------------------------------
#
#  fig, axarr = plt.subplots(1, 3, figsize=(8,8), sharey = 'row')
#
#  payoff_fee = axarr[0].imshow(eq_payoff_map, origin = 'lower', vmin = 0,
#              vmax = 1, extent =
#                   [0,cap_upper_bound,0,amount_upper_bound],
#                   aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
#  fig.colorbar(payoff_fee, ax=axarr[0])
#  axarr[0].set_ylabel('Fee Rate')
#
#  pop_fee = axarr[1].imshow(eq_equity_map, origin = 'lower', vmin = 0,
#              vmax = 1, extent =
#                   [0,cap_upper_bound,0,amount_upper_bound],
#                   aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
#  fig.colorbar(pop_fee, ax=axarr[1])
#  axarr[1].set_ylabel('Fee Rate')
#
#  sust_eq_fee = axarr[2].imshow(resilience_map, origin = 'lower', vmin = 0,
#                 vmax = 1, extent =
#                   [0,cap_upper_bound,0,amount_upper_bound],
#                   aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
#  fig.colorbar(sust_eq_fee, ax=axarr[2])
#  axarr[2].set_xlabel('Fee Threshold')
#  axarr[2].set_ylabel('Fee Rate')
#
#
#  fig.text(0.05,0.8, 'Industrial Profit', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')
#  fig.text(0.05,0.52, 'Population', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')
#  fig.text(0.05,0.2, 'Resilience', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')
#
#  plt.tight_layout()
#
#
#  payoff_max_fee = eq_payoff_fee>=tol * np.amax(eq_payoff_fee)
#  pop_max_fee = eq_pop_fee>=tol * np.amax(eq_pop_fee)
#  prob_max_fee = total_sust_eq_fee>=tol * np.amax(total_sust_eq_fee)
#
#  colormap_fee = np.zeros((50,50,4))
#
#  # Where we set the RGB for each pixel
#  colormap_fee[np.logical_and(payoff_max_fee, ~pop_max_fee)] = payoff_color
#  colormap_fee[np.logical_and(~payoff_max_fee, pop_max_fee)] = pop_color
#  colormap_fee[np.logical_and(~payoff_max_fee, ~pop_max_fee)] = [1,1,1,1]
#  colormap_fee[np.logical_and(payoff_max_fee, pop_max_fee)] = overlap_color
#  axarr2[1].imshow(colormap_fee, origin = 'lower', extent =
#                   [0,cap_upper_bound,0,amount_upper_bound],
#                   aspect = cap_upper_bound/amount_upper_bound)
#  axarr2[1].contour(prob_max_fee*1, [1], extent =
#                   [0,cap_upper_bound,0,amount_upper_bound], colors = 'k')
#  plt.legend(handles=patches, bbox_to_anchor=(1, 0.5), borderaxespad=0. )
#  plt.xlabel('Fee Threshold')
#  plt.ylabel('Fee Rate')

