import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

filename_filter = "Abschirmung"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
figure_path = current_path + "Abbildungen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}

################################################################################################



################################################################################################

data = datadict["1a"]

filename = "Dicke"

################################################################################################

R = data[1]
I = data[2]
R0 = R[0]
I0 = I[0]

xdata = data[0] / 20
ydata = np.log(R*I0 / (I*R0))
xerr = np.zeros(len(xdata))
yerr = np.sqrt( 1/R + 0.01**2 + 1/R0 + 0.01**2 )
yerr[0] = 0.00001

################################################################################################

def curve_func(x, a, b):
    return a*x + b

popt, pcov = curve_fit(f=curve_func, xdata=xdata[1:], ydata=ydata[1:], sigma=yerr[1:], absolute_sigma=True)
fit_values = popt
fit_value_errors = np.sqrt (np.diag(pcov) )
residuals = ydata[1:] - curve_func(xdata[1:], *popt)
chi_squared = np.sum((residuals / yerr[1:]) ** 2)

for j in range(len(popt)):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

fit_vals = np.linspace(0, np.max(xdata)*1.02, 300)

################################################################################################

plt.figure()

plt.grid()
plt.errorbar(xdata[1:], ydata[1:], xerr=xerr[1:], yerr=yerr[1:], fmt='o', label="Messdaten", color='b', ms=2, zorder=10, alpha=1)
plt.errorbar(xdata[0:1], ydata[0:1], xerr=xerr[0:1], yerr=yerr[0:1], fmt='o', label="Nicht gefitteter Messwert", color='r', ms=2, zorder=10, alpha=1)
plt.plot(fit_vals, curve_func(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared,1)}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend()

plt.title("Transmission gegen Absorberdicke")
plt.xlabel(r"Absorberdicke $d$ / $mm$")
plt.ylabel(r"Logarithmus der Transmission $\log{T}$")

plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)