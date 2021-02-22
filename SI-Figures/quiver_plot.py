import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from simulate import simulate_SES


#set resource parameters
r = 10
R_max = 100  # aquifer capacity

# Set industrial payoff parameters
a = 1000  # profit cap

b1 = 1
b2 = 1
q = -0.5
c = 0.8  # extraction cost parameter
d = 10

# Set domestic user parameters
k = 2 # water access parameter (how steeply benefit of water rises)
p = 3 # out-of-system wellbeing/payoff
h = 0.06 # rate at which wage declines with excess labor
g = 0.01 # rate at which wage increases with marginal benefit of more labor
m = 0.8 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
W_min = 0


dt = 1

N = 10

U_arr = np.linspace(1,55,N)
W_arr = np.linspace(1,30,N)

dU = np.zeros((N, N))
dW = np.zeros((N, N))
dR = np.zeros((N, N))

for j, U_0 in enumerate(U_arr):
  for l, W_0 in enumerate(W_arr):
    pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                    W_min, dt, 100, W_0, U_0)
    R, E, U, S, W, P, L, convergence = pp.run()
    dU[j,l] = U[-1] - U_0
    if U[-1] < 0.1:
     dW[j,l] = 0 - W_0
    else:
      dW[j,l] = (L[-1]/U[-1])*W[-1] - W_0


U, W = np.meshgrid(U_arr, W_arr)

plt.figure()
ax = plt.axes()
ax.quiver(U, W, dU, dW)

ax.set_xlabel('Population (U)')
ax.set_ylabel('Wage (W)')



