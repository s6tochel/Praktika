# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def save(filename):
    plt.savefig(os.path.join(dir_path, "figures", filename + ".png"), dpi=300)

# get directory path of current script
dir_path = os.path.dirname(os.path.realpath(__file__))

# ENTER DATA HERE
txtfile_name = "Schwellenspannung_Lebensdauer.txt"
save_name = "Schwellenspannung_Lebensdauer"
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

y= data[1]/data[2]
xerr = np.full(16, 50)

plt.scatter(data[0], y, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, s=30)
#Weg gelassen, da nicht sichtbar
#plt.errorbar(data[0], y, xerr=xerr, fmt='none', color=colors.pop(1), linewidth=2, zorder=2, alpha=1) 
plt.xlabel(r'Spannung $U$')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid()
plt.title(figure_title)

save(save_name)
