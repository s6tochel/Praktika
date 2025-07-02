import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad as integrate

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
fit_data_path = current_path + "Fit_data/"
data_path = parent_dir_path + "Daten/"
figure_path = parent_dir_path + "Abbildungen/"

################################################################################################

data1 = np.loadtxt(data_path + "Spektrum7.txt", delimiter="\t", dtype=float).T
data2 = np.loadtxt(data_path + "Spektrum10.txt", delimiter="\t", dtype=float).T

slice_ = (0, 750)
slice = range(slice_[0], slice_[1])

xdata = data1[0]
xerr = np.zeros(len(xdata))
N1, N2 = data1[1], data2[1]
N2[N2 == 0] = 1
ydata = N1 / N2
yerr = np.sqrt((N1 / (N2**2)) + (N1**2/(N2**3)))
yerr[yerr == 0] = 1

s_xdata = xdata[slice]
s_ydata = ydata[slice]
s_xerr = xerr[slice]
s_yerr = yerr[slice]

################################################################################################

# def fit_func(x, a, b):
#     return a*x + b

# popt, pcov = curve_fit(f=fit_func, xdata=xdata, ydata=ydata, sigma=yerr, absolute_sigma=True)
# fit_values = popt
# fit_value_errors = np.sqrt (np.diag(pcov))
# residuals = ydata - fit_func(xdata, *popt)
# chi_squared = np.sum((residuals / yerr) ** 2)
# ndof = len(ydata) - len(popt)
# red_chi_squared = chi_squared / ndof

# for j in range(len(popt)):
#     print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
# print(f"\tChi: {chi_squared}")
# print(f"\treduced Chi: {red_chi_squared}")

# fit_vals = np.linspace(0, np.max(xdata), 300)

################################################################################################

plt.figure()
plt.grid()

plt.errorbar(s_xdata, s_ydata, xerr=s_xerr, yerr=s_yerr, fmt='o', label=f'Messfehler', color='orange', ms=2, zorder=1, alpha=0.4)
plt.errorbar(s_xdata, s_ydata, fmt='o', label='Messwerte', color='g', ms=1, zorder=2, alpha=0.6)

plt.legend()

plt.title("CFD Schwelle Links")
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Quotient der Ticks $\frac{N_1}{N_2}$')

plt.savefig(figure_path + "SchwelleCFD2" + ".png", dpi=300)