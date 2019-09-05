import numpy as np
from scipy.optimize import minimize


def maximize(*args, **kwargs):
  neg_func = lambda *lambda_args: -args[0](*lambda_args)
  sol = minimize(neg_func, *(args[1:]), **kwargs)
  sol.fun = -sol.fun
  sol.jac = -sol.jac
  return sol



class simulate_SES:
  def __init__(self, r, R_0, R_max, a, b1, b2, q, c, d, p_cap, p_fine, fee, fee_cap, k, p, f, g, h,
               W_0, W_min, U_0, dt):
    self.r = r
    self.R_0 = R_0
    self.R_max = R_max
    self.a = a
    self.b1 = b1
    self.b2 = b2
    self.q = q
    self.c = c
    self.d = d
    self.p_cap = p_cap
    self.p_fine = p_fine
    self.k = k
    self.p = p
    self.f = f
    self.g = g
    self.h = h
    self.W_0 = W_0
    self.W_min = W_min
    self.U_0 = U_0
    self.dt = dt
    self.s = r/R_max
    self.fee = fee
    self.fee_cap = fee_cap


  def resource(self, R, E):
    return max(R + (-self.s*R + self.r - E)*self.dt, 0)


  def payoff(self, inputs, R, W):
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
# (self.a*np.sqrt((1 - np.exp(-self.b1*E))*(1 - np.exp(-self.b2*L)))


  def replicator(self, U, L, S, W):
    if U > 0:
      return max(U + U*self.h*(S*W*(L/U) - self.p)*self.dt, 0)
    else:
      return max(U + U*self.h*(-self.p)*self.dt, 0)


  def water_access(self, R):
    return (1 - np.exp(-self.k*R/self.R_max))/ (1 - np.exp(-self.k))
        ## OR -(1-x)**k + 1


  def wage_function(self, W, L, U, marg_benefit):
    if marg_benefit > 0.001:
      W += self.g*marg_benefit*self.dt
    else:
      W += -self.f*(U - L)*self.dt
    return np.clip(W, self.W_min, None)  # Lower bound on wage.


  def fine_amount(self, E):
    # policy cost is a function of extraction
    # if E > self.p_cap:
    #   F = self.p_fine
    # else:
    #   F = 0
    F = np.zeros(np.shape(E))
    F[E > self.p_cap] = self.p_fine
    return F

  def fee_amount(self, E):
    if E > self.fee_cap:
      fee = self.fee*(E - self.fee_cap)
    else:
      fee = 0
    return fee


  def multi_gradient_descent(self, U_avail, cost_args):
    R_avail = cost_args[0]
    W_t = cost_args[1]
    if R_avail < self.p_cap:
      # if resource is below threshold at which policy applies, only need to do
      # one optimization
      sol = maximize(self.payoff,
                     (R_avail/2, U_avail/2),
                     bounds = ((0, R_avail), (0, U_avail)),
                     args = cost_args)

    else:
      # Optimize below extraction threshold
      sol_1 = maximize(self.payoff,
                       (self.p_cap/2, U_avail/2),
                       bounds = ((0, self.p_cap), (0, U_avail)),
                       args = cost_args)

      # Optimize above extraction threshold
      sol_2 = maximize(self.payoff,
                       ((R_avail + self.p_cap)/2, U_avail/2),
                       bounds = ((self.p_cap, R_avail), (0, U_avail)),
                       args = cost_args)
      sol_2.fun -= self.p_fine

      # Take the best of the optimization solutions as the overall solution
      if sol_1.fun >= sol_2.fun:
        sol = sol_1
      else:
        sol = sol_2

    # Check if payoff is greater than 0 - keep if so, otherwise set to 0
    if sol.fun > 0:
      E_opt = sol.x[0]
      L_opt = sol.x[1]
      dual = sol.jac[1]
      payoff = sol.fun
    else:
      E_opt = 0
      L_opt = 0
      payoff = 0
      dual = -W_t

    return E_opt, L_opt, payoff, dual


  def run(self):
    max_dist = 100  # set distance to arbitrary (large) number
    tolerance = 0.1


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
      # Find W that maximizes payoff / minimizes cost, constrained by labor and
      # water available (labor available being that from the previous step).
      cost_args = (R_avail, W[t-1])

      E_opt, L_opt, payoff, dual = self.multi_gradient_descent(U[t-1], cost_args)

      # Store extraction / labor / payoff for E, L with max payoff.
      E.append(E_opt)
      L.append(L_opt)
      P.append(payoff)

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
        max_dist = max(np.linalg.norm(diff, axis = 1))
      if t == 2800:
        print('Exceeded maximum number of iterations')
        # Print initial conditions corresponding to failure to converge
        print('Fine:', self.p_fine)
        print('Fine Cap:', self.p_cap)
        print('Fee:', self.fee)
        print('Fee Cap:', self.fee_cap)
        print('R =',R[0])
        print('U =',U[0])
        print('W =',W[0])
        break

      t += 1
    return R, E, U, S, W, P, L
