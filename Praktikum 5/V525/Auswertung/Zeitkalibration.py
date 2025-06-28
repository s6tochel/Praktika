import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad as integrate

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = current_path + "Fit_data/"
figure_path = parent_dir_path + "Abbildungen/"

################################################################################################

data = np.loadtxt(data_path + "Spektrum12_b.txt", delimiter=" ", dtype=float).T

b0 = data[3]
b0_err = data[5]

dt = np.array( [32, 48, 64, 80, 96] ) -5

################################################################################################

xdata = dt
xerr = np.zeros(len(xdata))
ydata = b0
yerr = b0_err


################################################################################################

def fit_func(x, a, b):
    return a*x + b

popt, pcov = curve_fit(f=fit_func, xdata=xdata, ydata=ydata, sigma=yerr, absolute_sigma=True)
fit_values = popt
fit_value_errors = np.sqrt (np.diag(pcov))
residuals = ydata - fit_func(xdata, *popt)
chi_squared = np.sum((residuals / yerr) ** 2)
ndof = len(ydata) - len(popt)
red_chi_squared = chi_squared / ndof

for j in range(len(popt)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")
print(f"\treduced Chi: {red_chi_squared}")

fit_vals = np.linspace(0, np.max(xdata), 300)

################################################################################################

plt.figure()
plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label=f'Messfehler', color='blue', ms=2, zorder=1, alpha=1)
plt.plot(fit_vals, fit_func(fit_vals, *popt), label=r"Fit ($\chi_\text{}^2 \approx$" + f"{np.round(chi_squared, 3)})", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend()

plt.title("Zeitkalibration")
plt.xlabel(r'Zeitabstand des ns-Delay $\Delta t$')
plt.ylabel(r'Mittlere Kanalnummer der gefitteten Peaks $b_0$')

plt.savefig(figure_path + "Zeitkalibration" + ".png", dpi=300)