# Resilience and Equity in Social-Ecological Systems with Power Asymmetries

## Model Description
A stylized dynamical systems model of the interaction between communities, resource-based industries (such as agriculture or forestry), and natural resources in resource-based communities. Allows for exploring how different policies (a fine or a fee on industrial resource use) influence the community or industry and the overall resilience of the system.

## Usage
**Required Libraries:** NumPy, SciPy, MatPlotLib, Glob, CSV, pyDOE

**Running a single simulation:** After defining all of the parameters, run the following to simulate a single trajectory. If no policy (no fine or fee parameters) is specified, the default is no policy.
```
from simulate import simulate_SES
simulation = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, fine_cap, fine, fee_cap, fee)
R, E, U, S, W, P, L = simulation.run()
```
**Producing the colormaps:** Producing the colormaps involves two steps if running in parallel (highly recommended):
1) If running in parallel, run cm_parallel.py (will require the mpi4py library). This step takes a long time, and can be skipped if using the data in fee_colormap_data or fine_colormap_data.
2) Run restitch_colormap.py

All code for producing figures is in Figure Code.
