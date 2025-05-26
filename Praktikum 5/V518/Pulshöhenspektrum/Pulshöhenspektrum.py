# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import os

def save(filename):
    plt.savefig(os.path.join(dir_path, "/../Latex/figures/", filename + ".png"), dpi=300)

def landau_pdf(x, mpv, eta, A):
    # mpv: most probable value, eta: width, A: amplitude, C: offset
    # This is a simple approximation, not the true Landau!
    xi = (x - mpv) / eta
    return A * np.exp(-0.5 * (xi + np.exp(-xi))) 

def gaussian(x, a, b, c, d):
    return a * np.exp(-0.5 * ((x - b) / c)**2) + d

txtfile_name_1= "Pulshöhenspektrum_Unser.txt"
txtfile_name_2= "Pulshöhenspektrum_B109.txt"
save_name_1 = "Pulshöhenspektrum_Unser"
save_name_2 = "Pulshöhenspektrum_B109"
save_name_3 = "Pulshöhenspektrum_Landau"
xmin, xmax = 100, 800

# Construct full path to data file
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path_1 = os.path.join(dir_path, "..", "Messdaten", txtfile_name_1)
file_path_2 = os.path.join(dir_path, "..", "Messdaten", txtfile_name_2)
# Check if file exists
#print("Resolved file path:", file_path)
figure_title_1 = "Pulshöhenspektrum"
figure_title_2 = "Unser_Pulshöhenspektrum"

# Einlesen der Daten
data_1 = np.genfromtxt(file_path_1, dtype=int, autostrip=True).T
mask = (data_1[0] >= xmin) & (data_1[0] <= xmax)
data_2 = np.genfromtxt(file_path_2, dtype=int, autostrip=True).T
colors = ['black', 'red', 'purple', 'cyan','indigo', 'yellow']

channel_1 = data_1[0]
channel_2 = data_2[0]
N_1 = data_1[1]
N_1_err = np.sqrt(data_1[1])
#N_1[N_1 == 0] = 1e-6
N_2 = data_2[1]
N_2_err = np.sqrt(data_2[1])
#N_2[N_2 == 0] = 1e-6
channel_2_cut = data_2[0][mask]
N_2_cut = data_2[1][mask]
N_2_err_cut = N_2_err[mask]
N_2_err_cut[N_2_err_cut < 1e-4] = 1e-4

plt.errorbar(channel_1, N_1, yerr=N_1_err, fmt='o', label=f'Messfehler', color='orange', ms=2, zorder=1, alpha=0.8)
plt.errorbar(channel_1, N_1, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.8)
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid(True)
plt.title(figure_title_1)
save(dir_path + "/../Latex/figures/" + save_name_1)

plt.figure()
plt.errorbar(channel_2, N_2, yerr=N_2_err, fmt='o', label=f'Messfehler', color='orange', ms=2, zorder=1, alpha=0.8)
plt.errorbar(channel_2, N_2, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.8)
plt.legend()
plt.grid(True)
plt.title(figure_title_2)
save(dir_path + "/../Latex/figures/" + save_name_2)

valid = (N_2_cut > 0) & np.isfinite(N_2_cut) & np.isfinite(N_2_err_cut)
channel_2_cut_fit = channel_2_cut[valid]
N_2_cut_fit = N_2_cut[valid]
N_2_err_cut_fit = N_2_err_cut[valid]

popt, pcov = curve_fit(
    landau_pdf,
    channel_2_cut_fit, N_2_cut_fit,
    sigma=N_2_err_cut_fit,
    absolute_sigma=True,
    maxfev=100000,
    p0=[250, 30, 40],
    bounds=(
        [0, 1, 10],   # eta mindestens 1
        [np.inf, 200, np.inf]
    )
)
fit_values = popt
fit_value_errors = np.sqrt(np.diag(pcov))
residuals = N_2_cut_fit - landau_pdf(channel_2_cut_fit, *popt)
chi_squared = np.sum((residuals / N_2_err_cut_fit) ** 2)

popt2, pcov2 = curve_fit(
    gaussian,
    channel_2_cut_fit,
    N_2_cut_fit,
    sigma=N_2_err_cut_fit,
    absolute_sigma=True,
    maxfev=1000000,
    p0=[max(N_2_cut_fit), channel_2_cut_fit[np.argmax(N_2_cut_fit)], 30, 0]
)
fit_values2 = popt2
fit_value_errors2 = np.sqrt(np.diag(pcov2))
residuals2 = N_2_cut_fit - gaussian(channel_2_cut_fit, *popt2)
chi_squared2 = np.sum((residuals2 / N_2_err_cut_fit) ** 2)

for j in range(len(fit_values)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")
for j in range(len(fit_values2)):
    print(f"{j}:\t{fit_values2[j]}\t± {fit_value_errors2[j]}")
print(f"\tChi: {chi_squared2}")

plt.figure()
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, yerr=N_2_err_cut_fit, fmt='o', label='Messwerte', color='orange', ms=2)
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.8)
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid(True)
plt.title("Pulshöhenspektrum (Bereich 100–800)")
save(dir_path + "/../Latex/figures/" + save_name_3 + "_ohne_fit")

plt.figure()
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, yerr=N_2_err_cut_fit, fmt='o', label='Messwerte', color='orange', ms=2)
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.8)
x_fit = np.linspace(min(channel_2_cut_fit), max(channel_2_cut_fit), 300)
plt.plot(x_fit, landau_pdf(x_fit, *popt), color='black', label=r"Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")")
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid(True)
plt.title("Pulshöhenspektrum (Bereich 100–800) angepasst an eine Landauverteilung")
save(dir_path + "/../Latex/figures/" + save_name_3)

plt.figure()
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, yerr=N_2_err_cut_fit, fmt='o', label='Messwerte', color='orange', ms=2)
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.8)
x_fit = np.linspace(min(channel_2_cut_fit), max(channel_2_cut_fit), 300)
plt.plot(x_fit, gaussian(x_fit, *popt2), color='black', label=r"Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared2,0))}" + ")")
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid(True)
plt.title("Pulshöhenspektrum (Bereich 100–800) angepasst an eine Gaußverteilung")
save(dir_path + "/../Latex/figures/" + save_name_3 + "_gaussian")

plt.figure()
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, yerr=N_2_err_cut_fit, fmt='o', label='Messwerte', color='orange', ms=2, alpha=0.4)
plt.errorbar(channel_2_cut_fit, N_2_cut_fit, fmt='o', label='Messwerte', color='g', ms=2, zorder=2, alpha=0.4)
x_fit = np.linspace(min(channel_2_cut_fit), max(channel_2_cut_fit), 300)
plt.plot(x_fit, landau_pdf(x_fit, *popt), color='black', zorder=4, label=r"Landau-Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")")
plt.plot(x_fit, gaussian(x_fit, *popt2), color='red', zorder=3, label=r"Gaus-Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared2,0))}" + ")")
plt.xlabel(r'Kanalnummer $b$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid(True)
plt.title("Pulshöhenspektrum (Bereich 100–800) angepasst")
save(dir_path + "/../Latex/figures/" + save_name_3 + "_gemischt")