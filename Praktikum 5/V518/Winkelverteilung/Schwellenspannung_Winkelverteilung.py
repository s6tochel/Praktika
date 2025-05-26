# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from np_to_latex import to_latex_table

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

def save(filename):
    plt.savefig(os.path.join(dir_path, "/../Latex/figures/", filename + ".png"), dpi=300)

def comma_to_point(file1_name, file2_name, dir):
    file1_path = os.path.join(dir, file1_name)
    file2_path = os.path.join(dir, file2_name)
    with open(file1_path, "r") as file1, open(file2_path, "w") as file2:
        lines = [line.rstrip().replace(',', '.') + "\n" for line in file1]
        file2.writelines(lines)


# get directory path of current script
dir_path = os.path.dirname(os.path.realpath(__file__))
messwerte_path = os.path.normpath(os.path.join(dir_path, "../Messdaten"))

# ENTER DATA HERE
txtfile_name = "SchwellenspannungD12.txt"
new_txtfile_name = "SchwellenspannungD12_ctp.txt"
save_name = "Schwellenspannung_Winkelverteilung"
figure_title = "Schwellenkurve der Winkelverteilung"

# Construct full path to data file
comma_to_point(txtfile_name, new_txtfile_name, messwerte_path)
file_path = os.path.join(messwerte_path, new_txtfile_name)

# Load and plot data

data = np.loadtxt(file_path, skiprows=1).T
print(data)
colors = ['black', 'red', 'purple', 'cyan','indigo', 'yellow']

Nk, N25, U = data[5], data[2], -data[1]

y= Nk / N25
yerr = np.sqrt( (Nk / N25**2) + (Nk**2 / N25**3) )
yerr[yerr == 0] = 1

header = [r"$-U$ / mV", r"$N_\mathrm{Koin.}/N_{25}$", r"$\Delta (N_\mathrm{Koin.}/N_{25})$"]
colformats = ['.0f', 'sci2', 'sci2']
write_latex_table(
    os.path.join(dir_path, "Schwellenspannung_Winkelverteilung_Latex.txt"),
    [U, y, yerr],
    header=header,
    colformats=colformats
)

plt.errorbar(U, y, yerr=yerr, label="Messwerte", color=colors.pop(0), linewidth=0.3, zorder=3, alpha=1, fmt='o', ms=3)
#Weg gelassen, da nicht sichtbar
#plt.errorbar(data[0], y, xerr=xerr, fmt='none', color=colors.pop(1), linewidth=2, zorder=2, alpha=1) 
#konstante um wert zu markieren
plt.axvline(x=70, color='red', linestyle='-', linewidth=1)
plt.xlabel(r'Spannung $-U$ / mV')
plt.ylabel(r'Verhältnis der Zählraten $N_\mathrm{Koin.}/N_{25}$')
plt.legend()
plt.grid()
plt.title(figure_title)

save(dir_path + "/../Latex/figures/" + save_name)
