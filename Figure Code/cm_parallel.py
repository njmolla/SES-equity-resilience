import numpy as np
import csv
from mpi4py import MPI


from make_policy_colormap import make_fine_cm
from make_policy_colormap import make_fee_cm

# This code is for setting up the colormap to run in parallel

comm = MPI.COMM_WORLD

import sys
print('hello from rank %d of %d' % (comm.rank, comm.size))
sys.stdout.flush() # prints right away

if comm.size < 100:
  print('ERROR only running on %d processors' % comm.size)
  sys.exit()

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


# Step size
dt = 0.08

policy = 'fee'

n = 50 # number of points in colormap

# need number of processors to be a square
cells_per_processor = int(np.sqrt(n**2 / comm.size))
ppr = n / cells_per_processor # number of pieces per row (each processor runs one piece)
# Set policy parameters
if policy == 'fine':
  fine_caps_all  = np.linspace(0,10,n) # array of all fine thresholds
  fines_all = np.linspace(0,200,n) # array of all fine amounts
  fees = 0 # fee amount
  fee_caps = 200
  # break full arrays of policies into pieces for each processor
  fine_caps = fine_caps_all[int(cells_per_processor*(comm.rank%ppr)):int(cells_per_processor*(comm.rank%ppr)) + cells_per_processor]
  fines = fines_all[cells_per_processor*int(comm.rank/ppr):cells_per_processor*int(comm.rank/ppr) + cells_per_processor]
  eq_payoff, eq_pop, total_sust_eq =\
    make_fine_cm(fines, fine_caps, fees, fee_caps, r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, W_min, dt)
else:
  fine_caps  = 200 # threshold at which fine is applied
  fines = 0 # fine amount
  fee_caps_all = np.linspace(0, 12, n)
  fees_all = np.linspace(0, 100, n) * dollar_scale # fee amount
  # break full arrays of policies into pieces for each processor
  fee_caps = fee_caps_all[int(cells_per_processor*(comm.rank%ppr)):int(cells_per_processor*(comm.rank%ppr)) + cells_per_processor]
  fees = fees_all[cells_per_processor*int(comm.rank/ppr):cells_per_processor*int(comm.rank/ppr) + cells_per_processor]
  eq_payoff, eq_pop, total_sust_eq =\
    make_fee_cm(fines, fine_caps, fees, fee_caps, r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, W_min, dt)

# Save colormap data
with open('eq_payoff_%s.csv'%(comm.rank), 'w+') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerows(eq_payoff)

with open('eq_pop_%s.csv'%(comm.rank), 'w+') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerows(eq_pop)

with open('total_sust_eq_%s.csv'%(comm.rank), 'w+') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerows(total_sust_eq)

