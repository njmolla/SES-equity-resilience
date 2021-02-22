import numpy as np
import pyDOE as doe

import sys
sys.path.append('../')
from simulate import simulate_SES
import pickle
import bz2

def create_2d_list(row_length, num_rows):
  row = [0] * row_length
  x = [0] * num_rows
  for i in range(num_rows):
    x[i] = row[:]
  return x


def make_policy_cm(fines, fine_caps, fees, fee_caps, r, R_max, a, b1, b2, q, c, d,
                   k, p, h, g, m, W_min, dt, processor_num, cells_per_processor):
  """
  Produces a colormap for various combinations of fines and thresholds.
  For each policy, runs simulation over 100 different trajectories and averages
  the payoff and population and keeps track of the proportion sustainable equilibrium.
  """
  # check which policy is an array
  if hasattr(fines, "__len__"):
    amounts = fines
    caps = fine_caps
    policy = 'fine'
  else:
    amounts = fees
    caps = fee_caps
    policy = 'fee'

  # set initial conditions to loop over
  np.random.seed(1)
  num_points = 80
  initial_points = doe.lhs(3, samples = num_points)
  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20

  print('running')
#
#  Es = create_2d_list(len(amounts), len(caps))
#  Us = create_2d_list(len(amounts), len(caps))
#  Ps = create_2d_list(len(amounts), len(caps))
#  Ws = create_2d_list(len(amounts), len(caps))

  for i, cap in enumerate(caps):
    for j, amount in enumerate(amounts):
      R_trajectories = [0]*num_points
      U_trajectories = [0]*num_points
      P_trajectories = [0]*num_points
      W_trajectories = [0]*num_points
      L_trajectories = [0]*num_points

      for n, point in enumerate(initial_points):
        R_0 = point[0]
        U_0 = point[1]
        W_0 = point[2]
        if policy == 'fee':
          pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                            W_min, dt, R_0, W_0, U_0, fee_cap = cap, fee = amount)
        else:
          pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                            W_min, dt, R_0, W_0, U_0, fine_cap = cap, fine = amount)
        R, E, U, S, W, P, L, converged = pp.run()

        # save trajectories
        R_trajectories[n] = R
        U_trajectories[n] = U
        P_trajectories[n] = P
        W_trajectories[n] = W
        L_trajectories[n] = L

      # Save  compressed colormap data
      p_row = processor_num // 8
      p_column = processor_num % 8
      piece_number =  200*p_row + 5*p_column + 40*j + i#cells_per_processor*processor_num + i # goes from 0 to 1599
      with bz2.BZ2File("labor_%s.p"%(piece_number), 'w') as f:
        pickle.dump(L_trajectories, f)

      with bz2.BZ2File("pop_%s.p"%(piece_number), 'w') as f:
        pickle.dump(U_trajectories, f)

      with bz2.BZ2File("payoff_%s.p"%(piece_number), 'w') as f:
        pickle.dump(P_trajectories, f)

      with bz2.BZ2File("wage_%s.p"%(piece_number), 'w') as f:
        pickle.dump(W_trajectories, f)
