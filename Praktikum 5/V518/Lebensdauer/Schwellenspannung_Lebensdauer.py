# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def save(filename):
    plt.savefig(os.path.join(dir_path, "/../Latex/figures/", filename + ".png"), dpi=300)

def linear(x, a, b):
    return a * x + b
# get directory path of current script
dir_path = os.path.dirname(os.path.realpath(__file__))

# ENTER DATA HERE
txtfile_name = "Schwellenspannung_Lebensdauer.txt"
save_name = "Schwellenspannung_Lebensdauer"
save_name2 = "Schwellenspannung_Lebensdauer_ableitung"
save_name3 = "Schwellenspannung_Lebensdauer_fit"
figure_title = "Schwellenkurve der Lebensdauer"

# Construct full path to data file
file_path = os.path.join(dir_path, "Messwerte", txtfile_name)
print("Resolved file path:", file_path)

# Check if file exists
if not os.path.isfile(file_path):
    print(f"Fehler: Datei wurde nicht gefunden:\n{file_path}")
    exit()

# Load and plot data
data = np.genfromtxt(file_path, dtype=int, autostrip=True).T
colors = ['black', 'red', 'purple', 'cyan','indigo', 'yellow']
data[2][data[2] == 0] = 1e-6
N_mess = data[1]
N_mon = data[2]
y= N_mess/N_mon
##yerr = np.log(y * np.sqrt( (1/data[1]) + (1/data[2]) ) )
#print (yerr)
yerr = np.full(len(data[1]), 0.1)
print((y))
yerr = y * np.sqrt(1/N_mess + 1/N_mon)
print(yerr)
xerr = np.full(16, 50)
dy_dx = np.gradient(y, data[0])
d2y_dx2 = np.gradient(dy_dx, data[0])
d2y_dx2_err = np.zeros_like(y)
h = data[0][1] - data[0][0]  # Schrittweite

for i in range(1, len(y)-1):
    d2y_dx2_err[i] = (1/h**2) * np.sqrt(yerr[i+1]**2 + 4*yerr[i]**2 + yerr[i-1]**2)

dy_dx_err = np.zeros_like(dy_dx)

# Für die inneren Punkte (zentraler Differenzenquotient)
for i in range(1, len(y)-1):
    dx = data[0][i+1] - data[0][i-1]
    dy_dx_err[i] = np.sqrt(yerr[i+1]**2 + yerr[i-1]**2) / abs(dx)

# Für die Randpunkte (einseitige Differenz)
dy_dx_err[0] = np.sqrt(yerr[1]**2 + yerr[0]**2) / abs(data[0][1] - data[0][0])
dy_dx_err[-1] = np.sqrt(yerr[-1]**2 + yerr[-2]**2) / abs(data[0][-1] - data[0][-2])

max_idx = np.argmax(dy_dx)
wende_x = data[0][max_idx]
wende_y = y[max_idx]

plt.errorbar(data[0], y, yerr=yerr, xerr=xerr, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
plt.axvline(x=wende_x, color='red', linestyle='-', linewidth=1)
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'logarithmisch aufgetragen Verhältnis der Zählrate $N_\mathrm{mess}/N_\mathrm{monitor}$')
plt.legend()
plt.grid(True)
plt.title(figure_title)
save(dir_path + "/../Latex/figures/" + save_name)

plt.figure()
plt.errorbar(data[0], d2y_dx2, xerr=xerr, yerr=d2y_dx2_err, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'Ableitung $d(N_\mathrm{mess}/N_\mathrm{monitor})/dU$')
plt.legend()
plt.grid(True)
plt.title("Numerische Ableitung der Schwellenkurve")
save(dir_path + "/../Latex/figures/" + save_name2)

popt, pcov = curve_fit(f=linear, xdata=data[0], ydata=y, sigma=yerr, absolute_sigma=True, maxfev=10000)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = y - linear(data[0], *popt)
chi_squared = np.sum((residuals / yerr) ** 2)


for j in range(2):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

plt.figure()
plt.errorbar(data[0], y, yerr=yerr, xerr=xerr, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
x_fit = np.linspace(min(data[0]), max(data[0]), 300)
plt.plot(x_fit, linear(x_fit, *popt), color='black', label='Lineare Anpassung an das logarithmische Verhältnis', linewidth=0.3)
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'Verhältnis der Zählraten $N_\mathrm{Koin.}/N_{25}$')
plt.legend()
plt.grid(True)
plt.title(figure_title)

save(dir_path + "/../Latex/figures/" + save_name3)