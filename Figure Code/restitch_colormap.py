import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.patches as mpatches


def restitch_data(folder, cmap_type, num_points):
  """
  Combines the colormap data that was broken up for parallelization back into
  one single array. folder is the location of all of the colormap data, and the
  cmap_type is the keyword for the type of information being plotted, of which
  there are three types:
    equilibrium payoff for industrial users ("eq_payoff")
    equilibrium domestic user population ("eq_pop")
    proportion of sustainable equilibrium ("total_sust_eq")
  """
  num_processors = 100
  cells_per_processor = int(np.sqrt(num_points**2 / num_processors))
  ppr = round(num_points / cells_per_processor) # number of pieces per row
  cmap_arr = np.zeros((num_points, num_points))
  files = glob.glob(folder + cmap_type + '_*.csv')
  for file in files:
    n = int(file[len(folder + cmap_type + '_'):len(file)-4]) # processor number
    y_index = int(cells_per_processor * (n%ppr))
    x_index = cells_per_processor*int(n/ppr)
    data = np.loadtxt(file, delimiter = ',')
    x_length = np.shape(data)[0]
    if len(np.shape(data)) == 2:
      y_length = np.shape(data)[1]
      cmap_arr[x_index:x_index + x_length, y_index:y_index + y_length] = data
    else:
      cmap_arr[x_index:x_index + x_length, y_index] = data
  return cmap_arr

# Plot fine colormaps

policy = 'fine'
folder = policy +'_colormap_data\\'

eq_payoff_fine = restitch_data(folder, 'eq_payoff', 50)
eq_pop_fine = restitch_data(folder, 'eq_pop', 50)
total_sust_eq_fine = restitch_data(folder, 'total_sust_eq', 50)

# set extent of colormaps
cap_upper_bound = 10 # upper bound of fine thresholds
amount_upper_bound = 200 # upper bound of fine or fee amounts

fig, axarr = plt.subplots(3, 2, figsize=(8,8), sharex = 'col')
axarr[0,0].set_title('With a Fine', fontsize = 'x-large')
payoff_fine = axarr[0, 0].imshow(eq_payoff_fine, origin = 'lower', vmin = 0, vmax = 155, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
axarr[0,0].set_ylabel('Fine Amount')

pop_fine = axarr[1, 0].imshow(eq_pop_fine, origin = 'lower', vmin = 0, vmax = 12, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
axarr[1, 0].set_ylabel('Fine Amount')

sust_eq_fine = axarr[2, 0].imshow(total_sust_eq_fine, origin = 'lower', vmin = 0, vmax = 0.9, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
axarr[2,0].set_xlabel('Fine Threshold')
axarr[2,0].set_ylabel('Fine Amount')

plt.figure()
tol = 0.95
payoff_max_fine = eq_payoff_fine>=tol*np.amax(eq_payoff_fine)
pop_max_fine = eq_pop_fine>=tol*np.amax(eq_pop_fine)
prob_max_fine = total_sust_eq_fine>=tol*np.amax(total_sust_eq_fine)

im_payoff = plt.imshow(payoff_max_fine,origin = 'lower', extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'Reds', alpha = 0.7)
im_pop = plt.imshow(pop_max_fine, origin = 'lower', extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'Blues', alpha = 0.6)
plt.contour(prob_max_fine*1, [1], extent =
                 [0,cap_upper_bound,0,amount_upper_bound])

# get colors from colormap for legend
payoff_color = im_payoff.cmap(im_payoff.norm([0,1]))
pop_color = im_pop.cmap(im_pop.norm([0,1]))

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=payoff_color[1], label='"Good" outcome for industry', alpha = 0.5),
           mpatches.Patch(color=pop_color[1], label='"Good" outcome for the community', alpha = 0.6),
           mpatches.Patch(edgecolor = 'black', facecolor = 'white', linewidth = 1.5, label='"High" resilience')]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), borderaxespad=0. )
plt.xlabel('Cap')
plt.ylabel('Fine Amount')


# Plot fee colormaps

policy = 'fee'
folder = policy +'_colormap_data\\'

eq_payoff_fee = restitch_data(folder, 'eq_payoff', 50)
eq_pop_fee = restitch_data(folder, 'eq_pop', 50)
total_sust_eq_fee = restitch_data(folder, 'total_sust_eq', 50)


cap_upper_bound = 10
amount_upper_bound = 100

axarr[0,1].set_title('With a Fee', fontsize = 'x-large')
payoff_fine = axarr[0, 1].imshow(eq_payoff_fee, origin = 'lower', vmin = 0, vmax = 155, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
fig.colorbar(payoff_fine, ax=axarr[0, 1])
axarr[0,1].set_ylabel('Fee Rate')

axarr[1, 1].imshow(eq_pop_fee, origin = 'lower', vmin = 0, vmax = 12, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
fig.colorbar(pop_fine, ax=axarr[1, 1])
axarr[1,1].set_ylabel('Fee Rate')

axarr[2, 1].imshow(total_sust_eq_fee, origin = 'lower', vmin = 0, vmax = 0.9, extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'viridis')
fig.colorbar(sust_eq_fine, ax=axarr[2, 1])
axarr[2,1].set_xlabel('Fee Threshold')
axarr[2,1].set_ylabel('Fee Rate')
#plt.colorbar()
#axarr[2, 1].colorbar()

fig.text(0.05,0.8, 'Industrial Profit', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')
fig.text(0.05,0.52, 'Population', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')
fig.text(0.05,0.2, 'Resilience', ha='center', va='center', rotation = 'vertical', fontsize = 'x-large')

plt.tight_layout()


payoff_max_fee = eq_payoff_fee>=tol*np.amax(eq_payoff_fine)
pop_max_fee = eq_pop_fee>=tol*np.amax(eq_pop_fine)
prob_max_fee = total_sust_eq_fee>=tol*np.amax(total_sust_eq_fee)
plt.figure()
plt.imshow(payoff_max_fee,origin = 'lower', extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'Reds', alpha = 0.7)
plt.imshow(pop_max_fee, origin = 'lower', extent =
                 [0,cap_upper_bound,0,amount_upper_bound],
                 aspect = cap_upper_bound/amount_upper_bound, cmap = 'Blues', alpha = 0.6)

plt.contour(prob_max_fee*1,[1] , extent =
                 [0,cap_upper_bound,0,amount_upper_bound])

plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), borderaxespad=0. )
plt.xlabel('Fee Threshold')
plt.ylabel('Fee Rate')