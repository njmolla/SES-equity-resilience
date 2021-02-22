import numpy as np
from mpi4py import MPI
import sys

from make_policy_colormap import make_policy_cm


# This code is for setting up the colormap to run in parallel

comm = MPI.COMM_WORLD

"""
Need to specify whether running a fee or fine colormap.

Splits up colormap into num_processors pieces to run on the cluster, saves the outcomes for each
objective (population, profit, resilience) as well as convergence information in pickles
"""
# Set the policy
policy = 'fine'


# total number of processors
num_processors = 50

if comm.size != num_processors:
  print('ERROR running on %d processors' % comm.size)
  sys.exit()

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


# Step size
dt = 0.08

n = 50 # colormap will be nxn

# want number of processors to be a factor of n
cells_per_processor = int(n**2 / num_processors)
ppr = n / cells_per_processor # number of pieces per row (each processor runs one piece)
# Set policy parameters
if policy == 'fine':
  fine_caps_all  = np.linspace(0,10,n) # array of all fine thresholds
  fines_all = np.linspace(0,200,n) # array of all fine amounts
  fees = 0 # fee amount
  fee_caps = 200
  # break full arrays of policies into pieces for each processor
  fine_caps = fine_caps_all[int(cells_per_processor*(comm.rank%ppr)):int(cells_per_processor*(comm.rank%ppr)) + cells_per_processor]
  fines = fines_all[int(comm.rank/ppr):int(comm.rank/ppr) + 1]
  make_policy_cm(fines, fine_caps, fees, fee_caps, r, R_max, a, b1, b2, q, c, d,
                   k, p, h, g, m, W_min, dt, comm.rank, cells_per_processor)
else:
  fine_caps  = 200 # threshold at which fine is applied
  fines = 0 # fine amount
  fee_caps_all = np.linspace(0, 10, n)
  fees_all = np.linspace(0, 40, n) # fee amount
  # break full arrays of policies into pieces for each processor
  fee_caps = fee_caps_all[int(cells_per_processor*(comm.rank%ppr)):int(cells_per_processor*(comm.rank%ppr)) + cells_per_processor]
  fees = fees_all[int(comm.rank/ppr):int(comm.rank/ppr) + 1]
  make_policy_cm(fines, fine_caps, fees, fee_caps, r, R_max, a, b1, b2, q, c, d,
                   k, p, h, g, m, W_min, dt, comm.rank, cells_per_processor)
        

