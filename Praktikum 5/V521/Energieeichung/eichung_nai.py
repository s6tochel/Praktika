import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def linear(x, a, b):
    return a*x + b

E_lit = np.array([121.7817, 244.6974, 344.2785, 778.9045, 867.380, 1100.9206, 1408.013])
E_lit_err = np.array([0.0003, 0.0008, 0.0012, 0.0024, 0.005, 0.006, 0.003])

dir_path = os.path.dirname(os.path.realpath(__file__))
data_hpge = np.loadtxt(dir_path + "/data_hpge.txt", delimiter=" ").T
data_nai = np.loadtxt(dir_path + "/data_nai.txt", delimiter=" ").T

b = data_nai[2]
b_err = data_nai[3]

takeout_full = [0, 1, 2]

b = np.delete(b, takeout_full)
b_err = np.delete(b_err, takeout_full)

takeout_show = [-3]

b_no_fit = b[takeout_show[0]:takeout_show[-1]+1]
E_lit_no_fit = E_lit[takeout_show[0]:takeout_show[-1]+1]
b = np.delete(b, takeout_show)
E_lit = np.delete(E_lit, takeout_show)
b_err_no_fit = b_err[takeout_show[0]:takeout_show[-1]+1]
E_lit_err_no_fit = E_lit_err[takeout_show[0]:takeout_show[-1]+1]
b_err = np.delete(b_err, takeout_show)
E_lit_err = np.delete(E_lit_err, takeout_show)


plt.errorbar(E_lit, b, yerr=b_err, xerr=E_lit_err, fmt='o', label=f'Messwerte', color='b', ms=4, zorder=1, alpha=1)
plt.errorbar(E_lit_no_fit, b_no_fit, yerr=b_err_no_fit, xerr=E_lit_err_no_fit, fmt='o', label=f'Nicht gefittete Messwerte', color='g', ms=4, zorder=1, alpha=0.8)

popt, pcov = curve_fit(f=linear, xdata=E_lit, ydata=b, sigma=b_err, absolute_sigma=True)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = b - linear(E_lit, *popt)
chi_squared = np.sum((residuals / b_err) ** 2)

for j in range(2):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

fit_vals = np.linspace(0, 1500, 300)
plt.plot(fit_vals, linear(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)

plt.xlabel("Literaturwerte $E$ der Peak-Energien")
plt.ylabel("gemessene Mittelwerte $b_0$ der Peaks")
plt.title("Geradenfit zur Energieeichung mit NaI-Detektor")
plt.grid()
plt.legend()

plt.savefig(dir_path + "/figure3.png", dpi=300)