import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad as integrate

txtfile_title = "Lebensdauermessung.txt"

save_title = "Lebensdauermessung"
figure_title = r"Lebensdauermessung des $^{133}$Cs-Zustands"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/"
figure_path = parent_dir_path + "Abbildungen/"
fit_data_path = current_path + "Fit_data/"
data = np.loadtxt(data_path + txtfile_title, delimiter="\t", dtype=int).T

################################################################################################



################################################################################################

slice_ = (1300, 6000)


# I, t0, sigma, tau
bounds_i = [0, 1500, 800]
bounds_o = [1, 2000, 1100]

################################################################################################

xdata = data[0]
ydata = data[1]
xerr = np.zeros(len(xdata))
yerr = np.sqrt(ydata)
yerr[yerr == 0] = 1

s_xdata = xdata[slice_[0]: slice_[1]]
s_ydata = ydata[slice_[0]: slice_[1]]
s_xerr = xerr[slice_[0]: slice_[1]]
s_yerr = yerr[slice_[0]: slice_[1]]

################################################################################################

sq2 = np.sqrt(2)

def sq_exp(y):
    return np.exp(-y**2)

def error_func_scalar(x):
    result, _ = integrate(sq_exp, 0, x)
    return result

error_func = np.vectorize(error_func_scalar)

# def error_func(x):
#     return integrate(sq_exp, 0, x)

def a(sigma, tau, t0):
    return (sigma**2 + tau*t0) / (sq2 * sigma * tau)

def b(sigma, tau, t0, t):
    return (tau*(t - t0) - sigma**2) / (sq2 * sigma * tau)

def faltung(t, I, t0, tau):
    # sigma = 97.8114862864931
    sigma = 2.25 * 97.8114862864931
    return (I / 2*tau) * np.exp( (sigma**2 - 2*tau*(t - t0)) / (2*tau**2) ) * (error_func(a(sigma, tau, t0)) + error_func(b(sigma, tau, t0, t)))

popt, pcov = curve_fit(f=faltung, xdata=s_xdata, ydata=s_ydata, sigma=s_yerr, absolute_sigma=True, bounds = (bounds_i, bounds_o))
fit_values = popt
fit_value_errors = np.sqrt (np.diag(pcov))
residuals = s_ydata - faltung(s_xdata, *popt)
chi_squared = np.sum((residuals / s_yerr) ** 2)
ndof = len(s_ydata) - len(popt)
red_chi_squared = chi_squared / ndof

for j in range(len(popt)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")
print(f"\treduced Chi: {red_chi_squared}")

fit_vals = np.linspace(np.min(s_xdata), np.max(s_xdata), 300)

################################################################################################

plt.figure()
plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label=f'Messfehler', color='orange', ms=2, zorder=1, alpha=0.4)
plt.errorbar(xdata, ydata, fmt='o', label='Messwerte', color='g', ms=1, zorder=2, alpha=0.6)
plt.plot(fit_vals, faltung(fit_vals, *popt), label=r"Fit ($\chi_\text{red.}^2 \approx$" + f"{np.round(red_chi_squared, 2)})", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend()

plt.title("Fit für die Lebensdauermessung")
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')

plt.savefig(figure_path + "Lebensdauerfit" + ".png", dpi=300)