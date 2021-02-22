# Resilience and Equity in Resource Based Communities

A stylized dynamical systems model of the interaction between communities, resource-based industries (such as agriculture or forestry), and natural resources in resource-based communities. Allows for exploring how regulating industrial resource extraction influences the community or industry and the overall resilience of the system. The paper based on this analysis can be found [here](https://www.nature.com/articles/s43247-021-00093-y).

## Usage
**Required Libraries:** NumPy, SciPy, MatPlotLib, Glob, CSV, pyDOE, pickle, bz2, mpi4py

**Running a single simulation:** After defining all of the parameters, run the following to simulate a single trajectory. If no policy (no fine or fee parameters) is specified, the default is no policy.
```
from simulate import simulate_SES
simulation = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, fine_cap, fine, fee_cap, fee)
R, E, U, S, W, P, L, convergence = simulation.run()
```
All code for producing figures is in [Figure Code](https://github.com/njmolla/SES-equity-resilience/tree/master/Paper_Figure_Code).

**Producing the colormaps:** Producing the colormaps involves two steps if running in parallel (highly recommended):
1) If running in parallel, run cm_parallel.py (will require the mpi4py library). This step takes a long time, and can be skipped if using the data in [constructed_colormaps](https://github.com/njmolla/SES-equity-resilience/tree/master/Paper_Figure_Code/constructed_colormaps).
2) Run restitch_colormap.py
