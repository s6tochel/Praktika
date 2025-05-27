# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import os

def save(filename):
    plt.savefig(os.path.join(dir_path, "/../Latex/figures/", filename + ".png"), dpi=300)

txtfile_name = "Lebensdauer.txt"
save_name = "Lebensdauer"
figure_title = "Lebensdauer der Myonen"

counter = np.arange(1, 11)
counts = np.array([5035, 3001, 1873, 1191, 786, 531, 377, 281, 173, 118])
counts_err = np.sqrt(counts)  # Use sqrt(counts) as errors
start = 2330927
start_err = np.sqrt(start)  # Error for start
stop = 9371864
stop_err = np.sqrt(stop)  # Error for stop
print(f"Start counts: {start} ± {start_err}")
print(f"Stop counts: {stop} ± {stop_err}")
time = 598756
tot = np.sum(counts)
tot_err = np.sqrt(np.sum(counts_err**2))  # Error for total counts
print (f"Total counts: {tot} ± {tot_err}")
z1 = -115e-3
z2 = -105e-3

N1= start - tot
N1_err = np.sqrt(start_err**2 + tot_err**2)  # Error for N1
N2 = stop - start
N2_err = np.sqrt(stop_err**2 + start_err**2)  # Error for N2
print(f"N1: {N1} ± {N1_err}")
print(f"N2: {N2} ± {N2_err}")

delta_t = 1e-6  # in seconds
Nz = delta_t * N1 * N2 / time  
Nz_err = delta_t * np.sqrt((N1_err * N2)**2 + (N2_err * N1)**2) / time  # Error for Nz
print(f"Nz: {Nz} ± {Nz_err}")

counts_corr = counts - Nz
counts_corr_err = np.sqrt(counts_err**2 + Nz_err**2)  # Error for corrected counts
print(f"Counts corrected: {counts_corr} ± {counts_corr_err}")
##########################################################FIT##########################################################
def expdecay(x, a, tau, b):
    return a * np.exp(-x / tau) + b
# Fit the data

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path_1 = os.path.join(dir_path, "..", "Messdaten", txtfile_name)

popt, pcov = curve_fit(
    expdecay, counter, counts_corr,
    sigma=counts_corr_err,  # Use sqrt(counts) as errors
    absolute_sigma=True,
    maxfev=10000000,
    p0=[counts_corr[0] - counts_corr[-1], 2, counts_corr[-1]]
)
fit_values = popt
fit_value_errors = np.sqrt(np.diag(pcov))
residuals = counts_corr - expdecay(counter, *popt)
chi_squared = np.sum((residuals / counts_corr_err) ** 2)

popt2, pcov2 = curve_fit(
    expdecay, counter, counts,
    sigma=counts_err,  # Use sqrt(counts) as errors
    absolute_sigma=True,
    maxfev=10000000,
    p0=[counts[0] - counts[-1], 2, counts[-1]]
)
fit_values2 = popt2
fit_value_errors2 = np.sqrt(np.diag(pcov2))
residuals2 = counts - expdecay(counter, *popt2)
chi_squared2 = np.sum((residuals2 / counts_err) ** 2)

plt.errorbar(counter, counts, yerr=counts_err, fmt='o', label='Messwerte', color='blue', ms=3, zorder=3, alpha=1)
x_fit2 = np.linspace(min(counter), max(counter), 300)
plt.plot(x_fit2, expdecay(x_fit2, *popt2), color='black', label=r"Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared2,2))}" + ")")
plt.xlabel(r'Zählernummer $n$')
plt.ylabel(r'Zählrate $N_n$')
plt.legend()
plt.grid(True)
plt.title(figure_title)
save(dir_path + "/../Latex/figures/" + save_name)


plt.figure()
plt.errorbar(counter, counts_corr, yerr=counts_corr_err, fmt='o', label='Messwerte', color='blue', ms=3, zorder=3, alpha=1)
x_fit = np.linspace(min(counter), max(counter), 300)
plt.plot(x_fit, expdecay(x_fit, *popt), color='black', label=r"Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,2))}" + ")")
plt.xlabel(r'Zählernummer $n$')
plt.ylabel(r'Zählrate $N_n$')
plt.legend()
plt.grid(True)
plt.title(figure_title + " (mit Korrektur)")
save(dir_path + "/../Latex/figures/" + save_name + "_fit_korrektur")

for j in range(len(fit_values)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

for j in range(len(fit_values2)):
    print(f"{j}:\t{fit_values2[j]}\t± {fit_value_errors2[j]}")
print(f"\tChi: {chi_squared2}")