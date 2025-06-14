import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

filename_filter = "Dosimetrie2"

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

umrechnungsfaktor = 32.4 * 31556736 * 10**(-9) * 10**(-3)


################################################################################################

data = datadict["b"]

filename = "U2"

R = 0.1

################################################################################################

xdata = data[0]
ydata = data[1]/R / m * umrechnungsfaktor
xerr = data[0]*0.01
yerr = ydata * np.sqrt((data[2]/data[1])**2 + (dm/m)**2)

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
plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label="Mo U=35kV", color='b', ms=2, zorder=10, alpha=1)
plt.plot(fit_vals, curve_func(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared,1)}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)
plt.legend(loc="upper left")

plt.title("Äquivalentdosis gegen Röhrenspannung")
plt.xlabel(r"Emissionsstrom $I$ / mA")
plt.ylabel(r"Äquivalentdosis $H$ / kSv a$^{-1}$")

plt.tight_layout()
plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)