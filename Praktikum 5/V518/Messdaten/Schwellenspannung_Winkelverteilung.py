# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def save(filename):
    plt.savefig(os.path.join(dir_path, "figures", filename + ".png"), dpi=300)

def comma_to_point(file1_name, file2_name, dir):
    file1 = open(dir + file1_name, "r")
    file2 = open(dir + file2_name, "w")

    lines = [line.rstrip() for line in file1].copy()

    data_lenght = len(lines)

    for i in range(data_lenght):
        lines[i] = lines[i].replace(',', '.') + "\n"

    file2.writelines(lines) 
    file1.close()
    file2.close()

# get directory path of current script
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
messwerte_path = dir_path + "Messwerte/"

# ENTER DATA HERE
txtfile_name = "SchwellenspannungD12.txt"
new_txtfile_name = "SchwellenspannungD12_ctp.txt"
save_name = "Schwellenspannung_Winkelverteilung"
figure_title = "Schwellenkurve der Winkelverteilung"

# Construct full path to data file
comma_to_point(txtfile_name, new_txtfile_name, messwerte_path)
file_path = messwerte_path + new_txtfile_name

# Load and plot data

data = np.loadtxt(file_path, skiprows=1).T
print(data)
colors = ['black', 'red', 'purple', 'cyan','indigo', 'yellow']

Nk, N25, U = data[5], data[2], -data[1]

y= Nk / N25
yerr = np.sqrt( (Nk / N25**2) + (Nk**2 / N25**3) )
yerr[yerr == 0] = 1

plt.errorbar(U, y, yerr=yerr, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
#Weg gelassen, da nicht sichtbar
#plt.errorbar(data[0], y, xerr=xerr, fmt='none', color=colors.pop(1), linewidth=2, zorder=2, alpha=1) 
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'Ticks $N$')
plt.legend()
plt.grid()
plt.title(figure_title)

save(dir_path + "figures/" + save_name)
