import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import bz2
from restitch_colormap import fix_colormap
import sys
sys.path.append('../')
#from scipy import ndimage


def calculate_averages(cmap_type):
  num_points = 50
  t_lengths = np.arange(10,500,10)
  files = glob.glob(cmap_type + '_*.p')
  cmap_arr = np.zeros((len(t_lengths), num_points, num_points))
  for file in files:
    n = int(file[len(cmap_type + '_'):len(file)-2]) # piece number
    y_index = int(n/num_points)
    x_index = int(n%num_points)
    with bz2.BZ2File(file, 'rb') as f:
      data = pickle.load(f)
    for i,t_length in enumerate(t_lengths):
      value = avg_trajectories(data, t_length)
      cmap_arr[i, x_index, y_index] = value
  with open("temporal_data.p", 'wb') as f:
        pickle.dump(cmap_arr, f)


def calculate_overlap(resilience_map, payoff_map, equity_map, threshold, n):
  """
  Calculate overlap between resilience and equity
  """
  resilience_area = resilience_map >= threshold * np.amax(resilience_map)
  equity_area = equity_map >= threshold * np.amax(equity_map)
  payoff_area = payoff_map >= threshold * np.amax(payoff_map)

  resilience_and_payoff_area = np.logical_and(resilience_area, payoff_area)
  E_R_profit = (np.sum(np.logical_and(resilience_and_payoff_area, equity_area))
      / np.sum(resilience_and_payoff_area))
  E_R = np.sum(np.logical_and(resilience_area, equity_area)) / np.sum(resilience_area)

  overlap_1 = (E_R - np.sum(equity_area)/(2*n**2)) / (1 - np.sum(resilience_area)/(2*n**2))
  overlap_2 = (E_R_profit - np.sum(equity_area)/(2*n**2)) / (1 - np.sum(resilience_and_payoff_area)/(2*n**2))
  return overlap_1, overlap_2

t_lengths = np.arange(10,2000,10) * 0.04
thresholds = [0.97, 0.95, 0.9]

param = 'p'

#with open("parameter_experiments\\%s\\payoff_snapshots.p"%(param), 'rb') as f:
with open("payoff_snapshots.p", 'rb') as f:
  payoff_data = pickle.load(f)

payoff_data = payoff_data[1:]

#with open("parameter_experiments\\%s\\wellbeing_snapshots.p"%(param), 'rb') as f:
with open("wellbeing_snapshots.p", 'rb') as f:
  wellbeing_data = pickle.load(f)

n = 50
tol = 0.95
denom = np.zeros((len(t_lengths),1,1))
denom[:,0,0] = np.amax(np.amax(payoff_data + wellbeing_data,axis = 1),axis=1)
denom = np.broadcast_to(denom,(len(t_lengths),n,n))
equity_data = np.where(payoff_data<0.1,0,wellbeing_data/denom)

resilience_map = np.loadtxt("constructed_colormaps\\fee_resilience.csv", delimiter = ',')
#resilience_map = np.loadtxt("parameter_experiments\\%s\\resilience.csv"%(param), delimiter = ',')

#resilience_map = fix_colormap(np.loadtxt("parameter_experiments\\%s\\fee_resilience.csv"%(param), delimiter = ','))

overlaps_1 = np.zeros(len(t_lengths))
overlaps_2 = np.zeros(len(t_lengths))


plt.figure()
for i in range(len(t_lengths)):
  payoff_map = payoff_data[i]

  wellbeing_map = wellbeing_data[i]
  wellbeing_extension = np.broadcast_to(wellbeing_map[:,-1], (n,n))
  wellbeing_extended = np.zeros((n,2*n))
  wellbeing_extended[:n,:n] = wellbeing_map
#  wellbeing_extended[:n,n:2*n] = np.transpose(wellbeing_extension)

  resilience_extension = np.broadcast_to(resilience_map[:,-1], (n,n))
  resilience_extended = np.zeros((n,2*n))
  resilience_extended[:n,:n] = resilience_map
#  resilience_extended[:n,n:2*n] = np.transpose(resilience_extension)

  overlaps_1[i],overlaps_2[i]  = calculate_overlap(resilience_map, payoff_map, wellbeing_map, tol, n)
#plt.plot(t_lengths, overlaps_1,'.', overlaps_2, '.')

plt.plot(t_lengths[11:100], overlaps_1[11:100], '.', label = 'Equity alignment with Resilience')
plt.plot(t_lengths[11:100], overlaps_2[11:100], '.', label = 'Equity alignment with Resilience and Profit')
plt.legend()


#plt.plot(t_lengths[15:-50], overlaps_1[15:-50], '.', label = 'Resilience-Equity alignment')
#plt.plot(t_lengths[15:-50], overlaps_2[15:-50], '.', label = 'Combined resilience/profit-equity alignment')


plt.xlabel('Time', fontsize = 'x-large')
plt.ylabel('Alignment', fontsize = 'x-large')
#plt.title('%s = 15'%(param), fontsize = 'x-large')
plt.legend()

