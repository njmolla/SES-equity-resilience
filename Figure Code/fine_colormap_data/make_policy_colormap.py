import numpy as np
import pyDOE as doe

from simulate_as_class import simulate_SES

def make_fine_cm(fines, fine_caps, fee, fee_cap, r, R_max, a, b1, b2, q, c, d,
                   k, p, f, g, h, W_min, dt):
  # set initial conditions to loop over
#  initial_Rs = np.linspace(0,100,5)
#  initial_Us = np.linspace(0,40,5)
#  initial_Ws = np.linspace(0,20,5)
  np.random.seed(0)
  initial_points = doe.lhs(3, samples = 100)
  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20

  eq_payoff = np.zeros((len(fines), len(fine_caps)))
  eq_pop = np.zeros((len(fines), len(fine_caps)))

  total_sust_eq = np.zeros((len(fines), len(fine_caps)))

  for i, fine_cap in enumerate(fine_caps):
    for j, fine in enumerate(fines):

      for point in initial_points:
        R_0 = point[0]
        U_0 = point[1]
        W_0 = point[2]
        pp = simulate_SES(r, R_0, R_max, a, b1, b2, q, c, d, fine_cap, fine, fee, fee_cap, k, p, f, g, h,
                   W_0, W_min, U_0, dt)
        R, E, U, S, W, P, L = pp.run()

        eq_payoff[j,i] += P[-1]
        eq_pop[j,i] += U[-1]
        if R[-1] < 95:
          total_sust_eq[j,i] += 1

  eq_payoff /= 100
  eq_pop /= 100
  total_sust_eq /= 100

  return eq_payoff, eq_pop, total_sust_eq


def make_fee_cm(fine, fine_cap, fees, fee_caps, r, R_max, a, b1, b2, q, c, d,
                   k, p, f, g, h, W_min, dt):
  # set initial conditions to loop over
  np.random.seed(0)
  initial_points = doe.lhs(3, samples = 100)
  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20

  eq_payoff = np.zeros((len(fees), len(fee_caps)))
  eq_pop = np.zeros((len(fees), len(fee_caps)))

  total_sust_eq = np.zeros((len(fees), len(fee_caps)))

  for i, fee_cap in enumerate(fee_caps):
    for j, fee in enumerate(fees):

      for point in initial_points:
        R_0 = point[0]
        U_0 = point[1]
        W_0 = point[2]
        pp = simulate_SES(r, R_0, R_max, a, b1, b2, q, c, d, fine_cap, fine, fee, fee_cap, k, p, f, g, h,
                   W_0, W_min, U_0, dt)
        R, E, U, S, W, P, L = pp.run()

        eq_payoff[j,i] += P[-1]
        eq_pop[j,i] += U[-1]
        if R[-1] < 95:
          total_sust_eq[j,i] += 1

  eq_payoff /= 100
  eq_pop /= 100
  total_sust_eq /= 100

  return eq_payoff, eq_pop, total_sust_eq