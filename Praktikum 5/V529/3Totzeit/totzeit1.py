import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

filename_filter = "Totzeit"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
figure_path = current_path + "Abbildungen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}

################################################################################################



################################################################################################

data = datadict["1"]

filename = "Strom"

################################################################################################

xdata = data[0] * 0.001
ydata = data[1]
xerr = xdata * 0.01
yerr = np.sqrt(ydata)

################################################################################################

# def curve_func(x, a, b, c,d):
#     return a*(np.cos(b*x + c)**2)+d

# popt, pcov = curve_fit(f=curve_func, xdata=xdata, ydata=ydata, sigma=yerr, absolute_sigma=True)
# fit_values = popt
# fit_value_errors = np.sqrt (np.diag(pcov) )
# residuals = ydata - curve_func(xdata, *popt)
# chi_squared = np.sum((residuals / yerr) ** 2)

# for j in range(len(popt)):
#     print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
# print(f"\tChi: {chi_squared}")

# fit_vals = np.linspace(-3, 1, 300)

################################################################################################

plt.figure()

plt.grid()
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label="Messdaten", color='b', ms=2, zorder=10, alpha=1)
# plt.plot(fit_vals, curve_func(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared,1)}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.plot(np.ones(20)*(0.08), np.linspace(0, 9164, 20), color="g", label="Strom maximaler Zählrate")
plt.legend()

plt.title("Zählrate gegen Emissionsstrom")
plt.xlabel(r"Emissionsstrom $I$ / $mA$")
plt.ylabel(r"Zählrate $R$ / $s^{-1}$")

plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)