import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

filename_filter = "Abstand"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
figure_path = current_path + "Abbildungen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}

################################################################################################

V = 0.00012539

rho = 1.293 * (273 / 299) * (990 / 1013)

drho = rho * np.sqrt((3 / 990)**2 + (1.5/26)**2)

m = V * rho
dm = V* drho

################################################################################################

data = datadict["1"]

filename = "Cu"

################################################################################################

x = (data[0] - data[2] + 11)
xdata = 1 / x**2
ydata = data[1]
xerr = 1 / x**3
yerr = np.ones(len(ydata)) * 0.1

################################################################################################

def curve_func(x, a, b):
    return a*x + b

popt, pcov = curve_fit(f=curve_func, xdata=xdata, ydata=ydata, sigma=yerr, absolute_sigma=True)
fit_values = popt
fit_value_errors = np.sqrt (np.diag(pcov) )
residuals = ydata - curve_func(xdata, *popt)
chi_squared = np.sum((residuals / yerr) ** 2)

for j in range(len(popt)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

fit_vals = np.linspace(0, np.max(xdata)*1.03, 300)

################################################################################################

plt.figure()

plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label="Cu Messwerte", color='b', ms=2, zorder=10, alpha=1)
plt.plot(fit_vals, curve_func(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared,1)}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend()

plt.title("Äquivalentdosis gegen Abstand")
plt.xlabel(r"inverses Abstandsquadrat $s^{-2}$ / cm$^{-2}$")
plt.ylabel(r"Äquivalentdosis $H$ / mSv")

plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)