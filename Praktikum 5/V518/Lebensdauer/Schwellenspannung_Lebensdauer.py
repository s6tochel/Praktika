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
N_mess = data[1] * 0.1
N_mon = data[2] * 0.1
x = data[0] * 0.1
y= N_mess/N_mon 
yerr = np.full(len(data[1]), 0.1)
yerr = y * np.sqrt(1/N_mess + 1/N_mon)
xerr = np.full(16, 5)
dy_dx = np.gradient(y, x)
d2y_dx2 = np.gradient(dy_dx, x)
#noch bestimmen
d2y_dx2_err = np.zeros_like(y)
h = x[1] - data[0][0]  # Schrittweite

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


plt.errorbar(x, y, yerr=yerr, xerr=xerr, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'Verhältnis der Zählrate $N_\mathrm{mess}/N_\mathrm{monitor}$')
plt.legend()
plt.grid(True)
plt.title(figure_title)
save(dir_path + "/../Latex/figures/" + save_name)

plt.figure()
plt.errorbar(x, d2y_dx2, xerr=xerr, yerr=d2y_dx2_err, label="Messwerte", color="black", linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)

plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'2 fache Ableitung $d(N_\mathrm{mess}/N_\mathrm{monitor})/dU$')
plt.axvline(x=130, color='red', linestyle='-', linewidth=1)
plt.legend()
plt.grid(True)
plt.title("Numerische Ableitung der Schwellenkurve")
save(dir_path + "/../Latex/figures/" + save_name2)
