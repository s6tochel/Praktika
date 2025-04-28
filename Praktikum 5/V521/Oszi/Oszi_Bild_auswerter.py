import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def exp(x, A, tau, offset):
    return A * np.exp(- x / tau) + offset

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

data = np.loadtxt(dir_path + "scope_8.csv", delimiter=",").T
t = data[0] * 1000000
U = data[1] * 1000

#len t = 2000
range_ = (800, 1600)

t_slice = t[range_[0]:range_[1]]
U_slice = U[range_[0]:range_[1]]
U_err_slice = np.ones(U_slice.size) *1
print(U_err_slice.size)
print(U_slice.size)


t0_idx = np.where(t_slice == 0)[0][0]
end_idx = range_[1] - range_[0] - 1

popt, pcov = curve_fit(f=exp, xdata=t_slice[t0_idx: end_idx], ydata=U_slice[t0_idx: end_idx], sigma=U_err_slice[t0_idx: end_idx], absolute_sigma=True)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = U_slice[t0_idx: end_idx] - exp(t_slice[t0_idx: end_idx], *popt)
chi_squared = np.sum((residuals / U_err_slice[t0_idx: end_idx]) ** 2)

for j in range(3):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

fit_vals = np.linspace(t_slice[t0_idx], t_slice[end_idx], 300)
plt.plot(fit_vals, exp(fit_vals, *popt), label=f"Exp Fit", color="black", linewidth=1.7, zorder=3, alpha=1)

plt.scatter(t_slice, U_slice, s=4, zorder=2)
plt.xlabel(r'$t$ / $\mu$s')
plt.ylabel(r'$U$ / mV')
plt.grid()
plt.title("Oszillograph des Vorverstärkersignals mit NaI-Detektor")
plt.savefig(dir_path + "scope_8.png",  dpi=300)