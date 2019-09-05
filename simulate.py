import numpy as np
from scipy.optimize import minimize


def maximize(*args, **kwargs):
  neg_func = lambda *lambda_args: -args[0](*lambda_args)
  sol = minimize(neg_func, *(args[1:]), **kwargs)
  sol.fun = -sol.fun
  sol.jac = -sol.jac
  return sol



class simulate_SES:
  def __init__(self, r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
               W_min, dt, R_0, W_0, U_0, fine_cap = None, fine = 0, fee_cap = None, fee = 0):
    self.r = r
    self.R_0 = R_0
    self.R_max = R_max
    self.a = a
    self.b1 = b1
    self.b2 = b2
    self.q = q
    self.c = c
    self.d = d
    self.k = k
    self.p = p
    self.h = h
    self.g = g
    self.m = m
    self.W_0 = W_0
    self.W_min = W_min
    self.U_0 = U_0
    self.dt = dt
    self.s = r/R_max
    if fine_cap is None:
      self.fine_cap = R_max
    else:
      self.fine_cap = fine_cap
    self.fine = fine
    if fee_cap is None:
      self.fee_cap = R_max
    else:
      self.fee_cap = fee_cap
    self.fee = fee


  def resource(self, R, E):
    return max(R + (-self.s*R + self.r - E)*self.dt, 0)


  def profit(self, inputs, R, W):
    # Inputs should be an array with E and L as elements.
    E = inputs[0]
    L = inputs[1]
    if E <= 0 or L <= 0:
      try:
        return -self.c*(self.R_max - R)*E - W*L - self.a/self.d
      except:
        print('R:',R)
        print('E:', E)
        print('L:', L)
        print('W:', W)
        print('d:', self.d)
    else:
      return self.a*(self.b1*E**self.q + self.b2*L**self.q + 1)**(1/self.q) \
          - self.c*(self.R_max - R)*E - W*L - self.a/self.d - self.fee_amount(E)


  def replicator(self, U, L, S, W):
    if U > 0:
      return max(U + U*self.m*(S*W*(L/U) - self.p)*self.dt, 0)
    else:
      return max(U + U*self.m*(-self.p)*self.dt, 0)


  def water_access(self, R):
    return (1 - np.exp(-self.k*R/self.R_max))/ (1 - np.exp(-self.k))


  def wage_function(self, W, L, U, marg_benefit):
    if marg_benefit > 0.001:
      W += self.g*marg_benefit*self.dt
    else:
      W += -self.h*(U - L)*self.dt
    return np.clip(W, self.W_min, None)  # Lower bound on wage.


  def fine_amount(self, E):
    # policy cost is a function of extraction
    # if E > self.fine_cap:
    #   F = self.fine
    # else:
    #   F = 0
    F = np.zeros(np.shape(E))
    F[E > self.fine_cap] = self.fine
    return F

  def fee_amount(self, E):
    # if extraction is above threshold, fee amount is proportional to amount
    # above threshold
    if E > self.fee_cap:
      fee = self.fee*(E - self.fee_cap)
    else:
      fee = 0
    return fee


  def multi_gradient_descent(self, U_avail, cost_args):
    R_avail = cost_args[0]
    W_t = cost_args[1]
    if R_avail < self.fine_cap:
      # if resource is below threshold at which policy applies, only need to do
      # one optimization
      sol = maximize(self.profit,
                     (R_avail/2, U_avail/2),
                     bounds = ((0, R_avail), (0, U_avail)),
                     args = cost_args)

    else:
      # Optimize below extraction threshold
      sol_1 = maximize(self.profit,
                       (self.fine_cap/2, U_avail/2),
                       bounds = ((0, self.fine_cap), (0, U_avail)),
                       args = cost_args)

      # Optimize above extraction threshold
      sol_2 = maximize(self.profit,
                       ((R_avail + self.fine_cap)/2, U_avail/2),
                       bounds = ((self.fine_cap, R_avail), (0, U_avail)),
                       args = cost_args)
      sol_2.fun -= self.fine

      # Take the best of the optimization solutions as the overall solution
      if sol_1.fun >= sol_2.fun:
        sol = sol_1
      else:
        sol = sol_2

    # Check if profit is greater than 0 - keep if so, otherwise set to 0
    if sol.fun > 0:
      E_opt = sol.x[0]
      L_opt = sol.x[1]
      dual = sol.jac[1]
      profit = sol.fun
    else:
      E_opt = 0
      L_opt = 0
      profit = 0
      dual = -W_t

    return E_opt, L_opt, profit, dual


  def run(self):
    max_dist = 100  # set distance to arbitrary (large) number
    tolerance = 0.1 # tolerance for convergence


    # Initialize state variables.
    R = [self.R_0]  # initial resource state
    W = [self.W_0]  # initial wage
    U = [self.U_0]  # initial population; note that U[t] represents population at t+1

    # Initialize other variables.
    E = []
    L = []
    P = []
    S = []

    t = 1
    # start simulation
    while max_dist > tolerance*self.dt:
      # Calculate available resource for current time step (before extraction)
      R_avail = self.resource(R[t-1], 0)
      # Find W that maximizes profit / minimizes cost, constrained by labor and
      # water available (labor available being that from the previous step).
      cost_args = (R_avail, W[t-1])
      # Industrial user optimization
      E_opt, L_opt, profit, dual = self.multi_gradient_descent(U[t-1], cost_args)

      # Store extraction / labor / profit for E, L with max profit.
      E.append(E_opt)
      L.append(L_opt)
      P.append(profit)

      # Update state variables based on extraction / labor.
      R.append(self.resource(R[t-1], E[t-1]))
      S.append(self.water_access(R[t]))
      W.append(self.wage_function(W[t-1], L[t-1], U[t-1], dual))
      U.append(self.replicator(U[t-1], L[t-1], S[t-1], W[t]))

      if t > 10:
        # Take last ten points.
        states = np.transpose(np.array([R[-10::], U[-10::], W[-10::]]))
        # Subtract each point from the last point.
        diff = states[:9, :] - states[9, :]
        # Find the max Euclidean distance
        max_dist = max(np.linalg.norm(diff, axis = 1))
        
      # Break if exceeding maximum number of time steps
      if t == 2800:
        print('Exceeded maximum number of iterations')
        # Print initial conditions corresponding to failure to converge
        print('Fine:', self.fine)
        print('Fine Cap:', self.fine_cap)
        print('Fee:', self.fee)
        print('Fee Cap:', self.fee_cap)
        print('R =',R[0])
        print('U =',U[0])
        print('W =',W[0])
        break

      t += 1
    return R, E, U, S, W, P, L
