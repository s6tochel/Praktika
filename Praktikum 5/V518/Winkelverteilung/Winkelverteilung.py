import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
#from np_to_latex import to_latex_table

def cosine(x, a, b, c, d):
    return a * np.abs(np.cos(x + b))**c + d

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
data = np.loadtxt(dir_path + "/../Messdaten/Winkelverteilung.txt", delimiter=" ", skiprows=1).T

Winkel = data[0]
Winkel = np.deg2rad(Winkel)  # Grad in Radiant umwandeln
Koinzidenzen = data[1] 
Koinzidenzen_err = data[2] 
Koinzidenzen_err[Koinzidenzen_err == 0] = 1e-6

plt.figure()
plt.errorbar(Winkel, Koinzidenzen, yerr=Koinzidenzen_err, fmt='o', color='b', label='Messdaten')
plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Messdaten vor dem Fit")
plt.grid()
plt.legend()
plt.savefig(dir_path + "/../Latex/figures/WinkelverteilungPlot.png", dpi=300)

# Indizes, die ausgeschlossen werden sollen (Python zählt ab 0!)
ausschliessen = [2, 4, 5]

Winkel_aus = Winkel[ausschliessen]
Koinzidenzen_aus = Koinzidenzen[ausschliessen]
Koinzidenzen_err_aus = Koinzidenzen_err[ausschliessen]

Winkel = np.delete(Winkel, ausschliessen)
Koinzidenzen = np.delete(Koinzidenzen, ausschliessen)
Koinzidenzen_err = np.delete(Koinzidenzen_err, ausschliessen)

plt.figure()
plt.errorbar(Winkel, Koinzidenzen, yerr=Koinzidenzen_err, fmt='o', color='b', label='Messdaten')
plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Messdaten vor dem Fit")
plt.grid()
plt.legend()
plt.savefig(dir_path + "/../Latex/figures/WinkelverteilungPlot.png", dpi=300)

p0 = [4000, 0, 2, 100]

popt, pcov = curve_fit(f=cosine, xdata=Winkel, ydata=Koinzidenzen, sigma=Koinzidenzen_err, absolute_sigma=True, maxfev=1000000, p0=p0)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = Koinzidenzen - cosine(Winkel, *popt)
chi_squared = np.sum((residuals / Koinzidenzen_err) ** 2)


for j in range(3):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Koinzidenzen in Abhängigkeit vom Winkel")
#plt.errorbar(fit_values, cosine(fit_values, *popt), yerr=Koinzidenzen_err, fmt='o', label=r"Linearer Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")", color='g', ms=4, zorder=3, alpha=1)
plt.errorbar(np.rad2deg(Winkel_aus), Koinzidenzen_aus, yerr=Koinzidenzen_err_aus, fmt='x', color='r', alpha=0.75, label='ausgeschl. Werte')
plt.errorbar(np.rad2deg(Winkel), Koinzidenzen, yerr=Koinzidenzen_err, fmt='o', color='b', label='Messdaten')
x_fit = np.linspace(min(Winkel), max(Winkel), 300)
plt.plot(np.rad2deg(x_fit), cosine(x_fit, *popt), color='g', label=r"Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")")

plt.grid()
plt.legend()

plt.savefig(dir_path + "/../Latex/figures/cosAnpassung.png", dpi=300)