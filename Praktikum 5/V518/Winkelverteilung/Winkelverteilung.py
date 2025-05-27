import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
#from np_to_latex import to_latex_table

def sci_fmt(val, sig=2):
    return f"{val:.{sig}e}"

def write_latex_table(filename, columns, header=None, colformats=None):
    n_rows = len(columns[0])
    n_cols = len(columns)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{|" + "c|"*n_cols + "}\n")
        f.write(r"\hline" + "\n")
        if header:
            f.write(" & ".join(header) + r" \\ \hline" + "\n")
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                val = columns[j][i]
                if colformats and colformats[j]:
                    if colformats[j] == 'sci2':
                        row.append(f"{val:.2e}")
                    else:
                        row.append(f"{val:{colformats[j]}}")
                else:
                    row.append(str(val))
            f.write(" & ".join(row) + r" \\ \hline" + "\n")
        f.write(r"\end{tabular}" + "\n")

def cosine(x, a, b, c, d, m):
    return a * np.abs(np.cos(x + b))**c + m*x + d

def gaussian(x, a, b, c, d):
    return a * np.exp(-0.5 * ((x - b) / c)**2) + d

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
data = np.loadtxt(dir_path + "/../Messdaten/Winkelverteilung.txt", delimiter=" ", skiprows=1).T

Winkel_alt = data[0]
Winkel = np.deg2rad(Winkel_alt)  # Grad in Radiant umwandeln
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
Winkel_alt_aus = Winkel_alt[ausschliessen]
Koinzidenzen_aus = Koinzidenzen[ausschliessen]
Koinzidenzen_err_aus = Koinzidenzen_err[ausschliessen]

Winkel = np.delete(Winkel, ausschliessen)
Koinzidenzen = np.delete(Koinzidenzen, ausschliessen)
Koinzidenzen_err = np.delete(Koinzidenzen_err, ausschliessen)
Winkel_alt = np.delete(Winkel_alt, ausschliessen)

plt.figure()
plt.errorbar(Winkel, Koinzidenzen, yerr=Koinzidenzen_err, fmt='o', color='b', label='Messdaten')
plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Messdaten vor dem Fit")
plt.grid()
plt.legend()
plt.savefig(dir_path + "/../Latex/figures/WinkelverteilungPlot.png", dpi=300)

p0 = [4000, 0, 2, 100, 0]  # Startwerte für die Parameter [a, b, c, d, m]

popt, pcov = curve_fit(f=cosine, xdata=Winkel, ydata=Koinzidenzen, sigma=Koinzidenzen_err, absolute_sigma=True, maxfev=1000000, p0=p0)
fit_values = popt
fit_value_errors = np.sqrt(np.diag(pcov))
residuals = Koinzidenzen - cosine(Winkel, *popt)
chi_squared = np.sum((residuals / Koinzidenzen_err) ** 2)

popt2, pcov2 = curve_fit(f=gaussian, xdata=Winkel, ydata=Koinzidenzen, sigma=Koinzidenzen_err, absolute_sigma=True, maxfev=1000000, p0=[4000, 0, 2, 100])
fit_values2 = popt2
fit_value_errors2 = np.sqrt(np.diag(pcov2))
residuals2 = Koinzidenzen - gaussian(Winkel, *popt2)
chi_squared2 = np.sum((residuals2 / Koinzidenzen_err) ** 2)

header = [r"Winkel / $\degree$", r"$N_\mathrm{Koin.}$", r"$\Delta (N_\mathrm{Koin.})$"]
colformats = ['.0f', '.0f', '.0f']
write_latex_table(
    os.path.join(dir_path, "Winkelverteilung_Latex.txt"),
    [Winkel_alt, Koinzidenzen, Koinzidenzen_err],
    header=header,
    colformats=colformats
)

for j in range(len(fit_values)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

for j in range(len(fit_values2)):
    print(f"{j}:\t{fit_values2[j]}\t± {fit_value_errors2[j]}")
print(f"\tChi: {chi_squared2}")

plt.figure()
plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Koinzidenzen in Abhängigkeit vom Winkel angepasst an eine Cosinus Funktion")
plt.errorbar(np.rad2deg(Winkel_aus), Koinzidenzen_aus, yerr=2*Koinzidenzen_err_aus, fmt='x', color='r', alpha=0.75, label='ausgeschl. Werte', ms=4)
plt.errorbar(np.rad2deg(Winkel), Koinzidenzen, yerr=2*Koinzidenzen_err, fmt='o', color='orange', label='Messwerte', ms=4)
plt.errorbar(np.rad2deg(Winkel), Koinzidenzen, fmt='o', color='g', label='Messwerte', ms=4, zorder=2, alpha=0.8)
x_fit = np.linspace(min(Winkel), max(Winkel), 300)
plt.plot(np.rad2deg(x_fit), cosine(x_fit, *popt), color='black', label=r"Fit ($\chi^2 \approx $" + f"{(np.round(chi_squared,1))}" + ")")
plt.grid(True)
plt.legend()
plt.savefig(dir_path + "/../Latex/figures/cosAnpassung.png", dpi=300)


plt.figure()
plt.xlabel("Winkel $\\theta$ in $°$")
plt.ylabel("Koinzidenzen in $\\frac{1}{s}$")
plt.title("Koinzidenzen in Abhängigkeit vom Winkel angepasst an eine Gaußverteilung")
plt.errorbar(np.rad2deg(Winkel_aus), Koinzidenzen_aus, yerr=2*Koinzidenzen_err_aus, fmt='x', color='r', alpha=0.75, label='ausgeschl. Werte', ms=4)
plt.errorbar(np.rad2deg(Winkel), Koinzidenzen, yerr=2*Koinzidenzen_err, fmt='o', color='orange', label='Messwerte', ms=4)
plt.errorbar(np.rad2deg(Winkel), Koinzidenzen, fmt='o', color='g', label='Messwerte', ms=4, zorder=2, alpha=0.8)
x_fit = np.linspace(min(Winkel), max(Winkel), 300)
plt.plot(np.rad2deg(x_fit), gaussian(x_fit, *popt2), color='black', label=r"Fit ($\chi^2 \approx $" + f"{(np.round(chi_squared2,1))}" + ")")
plt.grid(True)
plt.legend()
plt.savefig(dir_path + "/../Latex/figures/gaussAnpassung.png", dpi=300)