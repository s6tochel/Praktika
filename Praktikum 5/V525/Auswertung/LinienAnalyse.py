# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from BoundMaker import bound_maker


# FUNCTIONS

def gauss(x, H1, H2, A, mu, sigma): 
    ground = (H1 + H2*x)
    gauss = A * np.exp( - 0.5 * ((x - mu) / sigma)**2)
    return ground + gauss

def double_gauss(x, H1, H2, A1, mu1, sigma1, A2, mu2, sigma2):
    ground = (H1 + H2*x)
    gauss1 = A1 * np.exp( - 0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp( - 0.5 * ((x - mu2) / sigma2)**2)
    return ground + gauss1 + gauss2

def triple_gauss(x, H1, H2, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    ground = (H1 + H2*x)
    gauss1 = A1 * np.exp( - 0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp( - 0.5 * ((x - mu2) / sigma2)**2)
    gauss3 = A3 * np.exp( - 0.5 * ((x - mu3) / sigma3)**2)
    return ground + gauss1 + gauss2 + gauss3

def quad_gauss(x, H1, H2, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
    ground = (H1 + H2*x)
    gauss1 = A1 * np.exp( - 0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp( - 0.5 * ((x - mu2) / sigma2)**2)
    gauss3 = A3 * np.exp( - 0.5 * ((x - mu3) / sigma3)**2)
    gauss4 = A4 * np.exp( - 0.5 * ((x - mu4) / sigma4)**2)
    return ground + gauss1 + gauss2 + gauss3 + gauss4


# GLOBAL VARIABLES

colors = ['black', 'red', 'blue', 'cyan', 'yellow']
chi_squared = 0

################################################################################################################################

# ENTER DATA HERE

txtfile_title = "Lebensdauermessung.txt"

save_title = "Lebensdauermessung"
figure_title = r"Lebensdauermessung des $^{133}$Cs-Zustands"

crop = False

ranges = []

gauss_fit_orders = []

# Bounds

H1_i = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
H1_o = [np.inf, np.inf, np.inf, np.inf, np.inf]
H2_i = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
H2_o = [np.inf, np.inf, np.inf, np.inf, np.inf]

A_i = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
A_o = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
x0_i = [2000, 3200, 4500, 5800, 7000, 1125, 1650]
x0_o = [2200, 3400, 4700, 5900, 7200, 1250, 1750]
sigma_i = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sigma_o = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]


################################################################################################################################

# get data from txt file
current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/"
figure_path = parent_dir_path + "Abbildungen/"
fit_data_path = current_path + "Fit_data/"
data = np.loadtxt(data_path + txtfile_title, delimiter="\t", dtype=int).T

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
I_err = np.sqrt(I)
I_err[I_err == 0] = 1

# create bounds
bounds = []
bounds = bound_maker(gauss_fit_orders, H1_i, H1_o, H2_i, H2_o, A_i, A_o, x0_i, x0_o, sigma_i, sigma_o)

# create txt file to save line data in
data_file = open(fit_data_path + save_title + ".txt", "w")
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
        param_len = 5
    elif gauss_fit_orders[i] == 2:
        popt, pcov = curve_fit(f=double_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
        param_len = 8
    elif gauss_fit_orders[i] == 3:
        popt, pcov = curve_fit(f=triple_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
        param_len = 11
    elif gauss_fit_orders[i] == 4:
        popt, pcov = curve_fit(f=quad_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
        param_len = 14
    else:
        print("\nGauss fit orders should either be 1, 2, 3 or 4!\n")
        raise ValueError

    gauss_values = popt
    gauss_value_errors = np.sqrt( np.diag(pcov) )

    # save fit values to peaks.txt file
    if gauss_fit_orders[i] == 1:
        residuals = I_slice - gauss(n_slice, *popt)
        data_file.write(f"{peak_nr} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]} {gauss_values[4]} {gauss_value_errors[4]}\n")
        peak_nr += 1

    if gauss_fit_orders[i] == 2:
        residuals = I_slice - double_gauss(n_slice, *popt)
        data_file.write(f"{peak_nr} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]} {gauss_values[4]} {gauss_value_errors[4]}\n")
        data_file.write(f"{peak_nr + 1} {gauss_values[5]} {gauss_value_errors[5]} {gauss_values[6]} {gauss_value_errors[6]} {gauss_values[7]} {gauss_value_errors[7]}\n")
        peak_nr += 2

    if gauss_fit_orders[i] == 3:
        residuals = I_slice - triple_gauss(n_slice, *popt)
        data_file.write(f"{peak_nr} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]} {gauss_values[4]} {gauss_value_errors[4]}\n")
        data_file.write(f"{peak_nr + 1} {gauss_values[5]} {gauss_value_errors[5]} {gauss_values[6]} {gauss_value_errors[6]} {gauss_values[7]} {gauss_value_errors[7]}\n")
        data_file.write(f"{peak_nr + 2} {gauss_values[8]} {gauss_value_errors[8]} {gauss_values[9]} {gauss_value_errors[9]} {gauss_values[10]} {gauss_value_errors[10]}\n")
        peak_nr += 3
    
    if gauss_fit_orders[i] == 4:
        residuals = I_slice - quad_gauss(n_slice, *popt)
        data_file.write(f"{peak_nr} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]} {gauss_values[4]} {gauss_value_errors[4]}\n")
        data_file.write(f"{peak_nr + 1} {gauss_values[5]} {gauss_value_errors[5]} {gauss_values[6]} {gauss_value_errors[6]} {gauss_values[7]} {gauss_value_errors[7]}\n")
        data_file.write(f"{peak_nr + 2} {gauss_values[8]} {gauss_value_errors[8]} {gauss_values[9]} {gauss_value_errors[9]} {gauss_values[10]} {gauss_value_errors[10]}\n")
        data_file.write(f"{peak_nr + 3} {gauss_values[11]} {gauss_value_errors[11]} {gauss_values[12]} {gauss_value_errors[12]} {gauss_values[13]} {gauss_value_errors[13]}\n")
        peak_nr += 4
    
    # print out chi squared values
    chi_squared = np.sum((residuals / I_err_slice) ** 2)
    ndof = len(I_slice) - len(popt)
    red_chi_squared = chi_squared / ndof

    print(f"Slice: {range_[0]} bis {range_[1]}")
    for j in range(param_len):
        print(f"{j}:\t{gauss_values[j]}\t± {gauss_value_errors[j]}")
    print(f"\tChi: {chi_squared}")
    print(f"\treduced Chi: {red_chi_squared}")

    # plot the gauss fits
    fit_vals = np.linspace(n_slice[0], n_slice[-1], 300)

    if gauss_fit_orders[i] == 1:
        plt.plot(fit_vals, gauss(fit_vals, *popt), label=f"Gauss {peak_nr-1}" + r" ($\chi_\text{red.}^2 \approx$" + f"{np.round(red_chi_squared, 2)})", color=colors.pop(0), linewidth=1.7, zorder=3, alpha=1)
    if gauss_fit_orders[i] == 2:
        plt.plot(fit_vals, double_gauss(fit_vals, *popt), label=f"Gauss {peak_nr-2},{peak_nr-1}" + r" ($\chi_\text{red.}^2 \approx$" + f"{np.round(red_chi_squared, 2)})", color=colors.pop(0), linewidth=1.7, zorder=3, alpha=1)
    if gauss_fit_orders[i] == 3:
        plt.plot(fit_vals, triple_gauss(fit_vals, *popt), label=f"Gauss {peak_nr-3},{peak_nr-2},{peak_nr-1}" + r" ($\chi_\text{red.}^2 \approx$" + f"{np.round(red_chi_squared, 2)})", color=colors.pop(0), linewidth=1.7, zorder=3, alpha=1)
    if gauss_fit_orders[i] == 4:
        plt.plot(fit_vals, quad_gauss(fit_vals, *popt), label=f"Gauss {peak_nr-4},{peak_nr-3},{peak_nr-2},{peak_nr-1}" + r" ($\chi_\text{red.}^2 \approx$" + f"{np.round(red_chi_squared, 2)})", color=colors.pop(0), linewidth=1.7, zorder=3, alpha=1)


data_file.close

if len(ranges) == 0:
    os.remove(fit_data_path + save_title + ".txt")

# plot the data
plt.errorbar(n, I, yerr=I_err, fmt='o', label=f'Messfehler', color='orange', ms=2, zorder=1, alpha=0.4)
plt.errorbar(n, I, fmt='o', label='Messwerte', color='g', ms=1, zorder=2, alpha=0.6)
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid()
plt.title(figure_title)

plt.savefig(figure_path + save_title + ".png", dpi=300)