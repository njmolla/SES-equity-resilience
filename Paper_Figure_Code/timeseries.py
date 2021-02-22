import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
sys.path.append('../')
from simulate import simulate_SES
# Plot trajectories for several different initial conditions

# Plot timeseries for a single run
#set resource parameters
r = 10
R_max = 100  # aquifer capacity

# Set industrial payoff parameters
a = 1000  # profit cap
#b1 = 0.05 # profit vs. water parameter
#b2 = 0.05 # profit vs. labor parameter
b1 = 1
b2 = 1
q = -0.5
c = 0.05 # extraction cost parameter
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

R_0 = 100
U_0 = 10
W_0 = 4

W_min = 0


ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                        W_min, dt, R_0, W_0, U_0)
R, E, U, S, W, P, L, convergence = ts.run()
R = np.array(R)

n = len(R)
T = n*dt
fig, axarr = plt.subplots(3)
axarr[0].plot(np.arange(n)*dt, R, color = 'k')
axarr[0].set_ylim([0, R_max])
axarr[0].set_title('Resource')
axarr[1].plot(np.arange(n)/(n/T), U, color = 'k')
axarr[1].set_title('Population')
axarr[1].set_ylim([0, 30])
eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color = 'k')
axarr[2].set_title('Effective Wage')
axarr[2].set_xlabel('Time')
plt.tight_layout()
patches = [mlines.Line2D([], [], color = 'k', linestyle = '-', linewidth=1, label='No Policy'),
           mlines.Line2D([], [], color = 'c', linestyle = '-', linewidth=1, label='Under a Fine')]

fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )


###########################################################
## set policy
fine = 130
fine_cap = 5

ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                        W_min, dt, R_0, W_0, U_0, fine_cap = fine_cap, fine = fine)
R, E, U, S, W, P, L, convergence = ts.run()

R = np.array(R)
n = len(R)


axarr[0].plot(np.arange(n)/(n/T), R[:n], color = 'c')
axarr[1].plot(np.arange(n)/(n/T), U[:n], color = 'c')
eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color = 'c')