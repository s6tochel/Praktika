import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from np_to_latex import to_latex_table

def linear(x, a, b):
    return a*x + b

E_lit_Eu = np.array([121.7817, 244.6974, 344.2785, 778.9045, 867.380, 964.057, 1085.837, 1112.076, 1408.013])
E_lit_Eu_err = np.array([0.0003, 0.0008, 0.0012, 0.0024, 0.003, 0.005, 0.010, 0.003, 0.003])
E_lit_Cs = np.array([661.7])
E_lit_Cs_err = np.zeros(len(E_lit_Cs))
E_lit_Co = np.array([1173.2, 1332.5])
E_lit_Co_err = np.zeros(len(E_lit_Co))

dir_path = os.path.dirname(os.path.realpath(__file__))
data_hpge = np.loadtxt(dir_path + "/data_hpge.txt", delimiter=" ").T
data_nai = np.loadtxt(dir_path + "/data_nai.txt", delimiter=" ").T

b = data_hpge[2]
b_err = data_hpge[3]

takeout_full = [6,7,8, -3]

b = np.delete(b, takeout_full)
b_err = np.delete(b_err, takeout_full)

takeout_show = []

# b_no_fit = b[takeout_show[0]:takeout_show[-1]+1]
# E_lit_no_fit = E_lit[takeout_show[0]:takeout_show[-1]+1]
# b = np.delete(b, takeout_show)
# E_lit = np.delete(E_lit, takeout_show)
# b_err_no_fit = b_err[takeout_show[0]:takeout_show[-1]+1]
# E_lit_err_no_fit = E_lit_err[takeout_show[0]:takeout_show[-1]+1]
# b_err = np.delete(b_err, takeout_show)
# E_lit_err = np.delete(E_lit_err, takeout_show)


plt.errorbar(E_lit_Eu, b[3:], yerr=b_err[3:], xerr=E_lit_Eu_err, fmt='o', label=f'Eu Messwerte', color='g', ms=4, zorder=3, alpha=1)
plt.errorbar(E_lit_Cs, b[0:1], yerr=b_err[0:1], xerr=E_lit_Cs_err, fmt='o', label=f'Cs Messwert', color='b', ms=4, zorder=3, alpha=1)
plt.errorbar(E_lit_Co, b[1:3], yerr=b_err[1:3], xerr=E_lit_Co_err, fmt='o', label=f'Co Messwerte', color='r', ms=4, zorder=3, alpha=1)

E_lit = np.concatenate((E_lit_Cs, E_lit_Co, E_lit_Eu))
E_lit_err = np.concatenate((E_lit_Cs_err, E_lit_Co_err, E_lit_Eu_err))

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
plt.title("Geradenfit zur Energieeichung mit HPGe-Detektor")
plt.grid()
plt.legend()

plt.savefig(dir_path + "/figure4.png", dpi=300)

to_latex_table([b, b_err, E_lit, E_lit_err], dir_path + "/hpge_tabelle.txt", round_to=[3, 3, 4, 4])