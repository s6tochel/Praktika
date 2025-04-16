import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ReadIn import readin, bound_maker

Material = "Au"

crop_i = 20
crop_o = 200
crop = True

ranges = [(90, 143), (143, 190)]
gauss_fits = [2, 2]

bounds_on = True
H_i = [1, 1]
H_o = [10, 10]
A_i = [150, 300, 200, 10]
A_o = [259, 500, 350, 50]
x0_i = [100, 125, 150, 175]
x0_o = [125, 140, 170, 190]
sigma_i = [0, 0, 0, 0]
sigma_o = [np.inf, np.inf, np.inf, np.inf]

error = 7

data = readin("Teil2/" + Material + ".txt")

n = data[0]
I = data[1]

if crop:
    n = n[crop_i:crop_o]
    I = I[crop_i:crop_o]
else:
    crop_i = 0

I_err = np.ones(len(I)) * error

bounds = []
if bounds_on:
    bounds = bound_maker(gauss_fits, H_i, H_o, A_i, A_o, x0_i, x0_o, sigma_i, sigma_o)

colors = ['red', 'green', 'purple', 'cyan','indigo', 'yellow']


# Lineare Funktion definieren
def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def double_gauss(x, H, A1, x01, sigma1, A2, x02, sigma2):
    return_val = H + A1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2))
    return return_val

data_file = open("peaks.txt", "a")
peak_nr = 1

for i in range(len(ranges)):
    range_ = ranges[i]
    param_len = 0

    n_slice = n[range_[0]-crop_i:range_[1]-crop_i]
    I_slice = I[range_[0]-crop_i:range_[1]-crop_i]
    I_err_slice = I_err[range_[0]-crop_i:range_[1]-crop_i]

    if not bounds_on:
        if gauss_fits[i] == 1:
            popt, pcov = curve_fit(f=gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True)
            param_len = 4
        if gauss_fits[i] == 2:
            popt, pcov = curve_fit(f=double_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True)
            param_len = 7
    else:
        bounds_i = bounds[i]
        if gauss_fits[i] == 1:
            popt, pcov = curve_fit(f=gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
            param_len = 4
        if gauss_fits[i] == 2:
            popt, pcov = curve_fit(f=double_gauss, xdata=n_slice, ydata=I_slice, sigma=I_err_slice, absolute_sigma=True, bounds=bounds_i)
            param_len = 7

    gauss_values = popt
    gauss_value_errors = np.diag(pcov)


    if gauss_fits[i] == 1:
        residuals = I_slice - gauss(n_slice, *popt)

        data_file.write(f"{Material} {peak_nr} {gauss_values[1]} {gauss_value_errors[1]} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]}\n")
        peak_nr += 1

    if gauss_fits[i] == 2:
        residuals = I_slice - double_gauss(n_slice, *popt)

        data_file.write(f"{Material} {peak_nr} {gauss_values[1]} {gauss_value_errors[1]} {gauss_values[2]} {gauss_value_errors[2]} {gauss_values[3]} {gauss_value_errors[3]}\n")
        data_file.write(f"{Material} {peak_nr + 1} {gauss_values[4]} {gauss_value_errors[4]} {gauss_values[5]} {gauss_value_errors[5]} {gauss_values[6]} {gauss_value_errors[6]}\n")
        peak_nr += 2

    chi_squared = np.sum((residuals / I_err_slice) ** 2)

    print(f"Slice: {range_[0]} bis {range_[1]}")
    for j in range(param_len):
        print(f"{j}:\t{gauss_values[j]}\t± {gauss_value_errors[j]}")
    print(f"\tChi: {chi_squared}")

    fit_vals = np.linspace(n_slice[0], n_slice[-1], 300)

    if gauss_fits[i] == 1:
        plt.plot(fit_vals, gauss(fit_vals, *popt), label=f'Gauss-Fit ({peak_nr-1})', color=colors.pop(0), linewidth=1.5, zorder=3, alpha=0.85)
    if gauss_fits[i] == 2:
        plt.plot(fit_vals, double_gauss(fit_vals, *popt), label=f'Doppel Gauss-Fit ({peak_nr-2}&{peak_nr-1})', color=colors.pop(0), linewidth=1.5, zorder=3, alpha=0.85)

data_file.close
# Daten und Fit plotten
plt.errorbar(n, I, yerr=I_err, fmt='o', label=f'Messfehler ($\sigma_R$ = {error})', color='orange', ms=2, zorder=1, alpha=0.6)
plt.errorbar(n, I, fmt='o', label='Messwerte', color='blue', ms=2, zorder=2)
# plt.plot(n, I)
plt.xlabel(r'Kanalnummer $n$')
plt.ylabel(r'Zählrate $R$ / $\text{s}^{-1}$')
plt.legend()
plt.grid()
plt.title(Material)

plt.savefig("figures/" + Material + ".png", dpi=300)