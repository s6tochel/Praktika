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

data1 = np.loadtxt(data_path + "Spektrum6.txt", delimiter=" ", dtype=float)
data2 = np.loadtxt(data_path + "Spektrum7_a.txt", delimiter=" ", dtype=float)
data3 = np.loadtxt(data_path + "Spektrum7_b.txt", delimiter=" ", dtype=float)

data = np.vstack((data1, data2, data3)).T

b0 = data[3]
b0_err = data[5]

delete_idx_list = [1, 4, 5, 7, 8]

b0 = np.delete(b0, delete_idx_list)
b0_err = np.delete(b0_err, delete_idx_list)

print(len(b0))

E_lit = np.array( [510.9989, 30.625, 30.973, 81, 302.9, 356, 383.8] )

################################################################################################

xdata = E_lit
xerr = np.zeros(len(E_lit))
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

fit_vals = np.linspace(np.min(xdata), np.max(xdata), 300)

################################################################################################

plt.figure()
plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label=f'Messfehler', color='blue', ms=2, zorder=1, alpha=1)
plt.plot(fit_vals, fit_func(fit_vals, *popt), label=r"Fit ($\chi_\text{}^2 \approx$" + f"{np.round(chi_squared, 2)})", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend()

plt.title("Energieeichung der linken Seite")
plt.xlabel(r'Literaturwerte der Energie $E_\text{lit.}$ / keV')
plt.ylabel(r'Mittlere Kanalnummer der gefitteten Peaks $b_0$')

plt.savefig(figure_path + "Energieeichung2" + ".png", dpi=300)