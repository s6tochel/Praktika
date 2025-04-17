# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from BoundMaker import bound_maker


# FUNCTIONS

def save(filename):
    plt.savefig(dir_path + "/figures/" + filename + ".png", dpi=300)

def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def double_gauss(x, H, A1, x01, sigma1, A2, x02, sigma2):
    return_val = H + A1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2))
    return return_val

# GLOBAL VARIABLES

colors = ['red', 'green', 'purple', 'cyan','indigo', 'yellow']

################################################################################################################################

# ENTER DATA HERE

txtfile_name = "spectrum.txt"

save_name = "spectrum_line56"

crop = (2700, 4500)

error = 25

ranges = [(2800, 4500)]
gauss_fit_orders = [2]

# Bounds

H_i = [0]
H_o = [40]
A_i = [0, 50]
A_o = [40, 125]
x0_i = [3100, 3750]
x0_o = [3500, 4250]
sigma_i = [0, 0]
sigma_o = [np.inf, np.inf]


################################################################################################################################

# get data from txt file
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = dir_path + "/" + txtfile_name
data = np.loadtxt(file_path, delimiter="\t", dtype=int).T

# define variables n (idices of bins), I (number of mesured photons in said bin), I_err (error on I)
# also crop range of data
if (isinstance(crop, tuple)) and (len(crop) == 2) and (crop[0] < crop[1]):
    n = data[0][crop[0]:crop[1]]
    I = data[1][crop[0]:crop[1]]
    crop_i = crop[0]
elif crop == False:
    n = data[0]
    I = data[1]
    crop_i = 0
else:
    print("\ncrop should either be a tuple containing a lower bound and an upper bound or set to False\n")
    raise TypeError
I_err = np.ones(len(I)) * error

# create bounds
bounds = []
bounds = bound_maker(gauss_fit_orders, H_i, H_o, A_i, A_o, x0_i, x0_o, sigma_i, sigma_o)

# create txt file to save line data in
data_file = open(dir_path + "/line_data/" + save_name + ".txt", "w")
peak_nr = 1

# actual calculations, iterate over ranges
for i in range(len(ranges)):
    range_ = ranges[i]
    param_len = 0

    # get slices of data
    n_slice = n[range_[0]-crop_i:range_[1]-crop_i]
    I_slice = I[range_[0]-crop_i:range_[1]-crop_i]
    I_err_slice = I_err[range_[0]-crop_i:range_[1]-crop_i]

    # do gauss fits
    bounds_i = bounds[i]
    if gauss_fit_orders[i] == 1:
        popt, pcov = curve_fit(f=gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
        param_len = 4
    elif gauss_fit_orders[i] == 2:
        popt, pcov = curve_fit(f=double_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
        param_len = 7
    else:
        print("\nGauss fit orders should either be 1 or 2\n")
        raise ValueError

    gauss_values = popt
    gauss_value_errors = np.diag(pcov)

    # save fit values to peaks.txt file
    if gauss_fit_orders[i] == 1:
        residuals = I_slice - gauss(n_slice, *popt)

        data_file.write(f"{save_name} {peak_nr} {gauss_values[1]} {gauss_value_errors[1]} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]}\n")
        peak_nr += 1
    if gauss_fit_orders[i] == 2:
        residuals = I_slice - double_gauss(n_slice, *popt)

        data_file.write(f"{save_name} {peak_nr} {gauss_values[1]} {gauss_value_errors[1]} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]}\n")
        data_file.write(f"{save_name} {peak_nr + 1} {gauss_values[4]} {gauss_value_errors[4]} {gauss_values[5]} {gauss_value_errors[5]} {gauss_values[6]} {gauss_value_errors[6]}\n")
        peak_nr += 2
    
    # print out chi squared values
    chi_squared = np.sum((residuals / I_err_slice) ** 2)

    print(f"Slice: {range_[0]} bis {range_[1]}")
    for j in range(param_len):
        print(f"{j}:\t{gauss_values[j]}\t± {gauss_value_errors[j]}")
    print(f"\tChi: {chi_squared}")

    # plot the gauss fits
    fit_vals = np.linspace(n_slice[0], n_slice[-1], 300)

    if gauss_fit_orders[i] == 1:
        plt.plot(fit_vals, gauss(fit_vals, *popt), label=f'Gauss-Fit ({peak_nr-1})', color=colors.pop(0), linewidth=1.5, zorder=3, alpha=0.85)
    if gauss_fit_orders[i] == 2:
        plt.plot(fit_vals, double_gauss(fit_vals, *popt), label=f'Doppel Gauss-Fit ({peak_nr-2}&{peak_nr-1})', color=colors.pop(0), linewidth=1.5, zorder=3, alpha=0.85)

data_file.close

# plot the data
plt.errorbar(n, I, yerr=I_err, fmt='o', label=f'Messfehler ($\sigma_R$ = {error})', color='orange', ms=2, zorder=1, alpha=0.6)
plt.errorbar(n, I, fmt='o', label='Messwerte', color='blue', ms=2, zorder=2)
plt.xlabel(r'Kanalnummer $n$')
plt.ylabel(r'Zählrate $R$ / $\text{s}^{-1}$')
plt.legend()
plt.grid()
plt.title(save_name)

save(save_name)