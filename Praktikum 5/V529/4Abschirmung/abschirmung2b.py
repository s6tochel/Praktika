import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from np_to_latex import to_latex_table

filename_filter = "Abschirmung"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
figure_path = current_path + "Abbildungen/"
table_path = current_path + "LatexTabellen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}

################################################################################################



################################################################################################

data1 = datadict["2a"]
data = datadict["2b"]

filename = "MaterialFilter"

################################################################################################

mu = -0.6554234886530051
dmu = 0.014601988902479338

R = data[1]
I = data[2]
R0 = data1[1][0]
I0 = data1[2][0]

print(I)
I[1] = 0.4
print(f"R = ")
print(R)

xdata = data[0] / 20
ydata = np.log(R*I0 / (I*R0))
xerr = np.zeros(len(xdata))
yerr = np.sqrt( 1/R + 0.01**2 + 1/R0 + 0.01**2 )

d = ydata[1] / mu
d_err = d * np.sqrt((yerr[1]/ydata[1])**2 + (dmu/mu)**2)

mus = -ydata / d
mus_err = mus * np.sqrt((yerr/ydata)**2 + (d_err/d)**2)
mus_err[1] = dmu

to_latex_table([ydata, yerr], table_path+"T2.txt", round_to=3)
to_latex_table([mus, mus_err], table_path+"mus2.txt", round_to=3)

print(ydata[1])
print(yerr[1])
print(d)
print(d_err)

################################################################################################

# def curve_func(x, a, b):
#     return a*x + b

# popt, pcov = curve_fit(f=curve_func, xdata=xdata[1:], ydata=ydata[1:], sigma=yerr[1:], absolute_sigma=True)
# fit_values = popt
# fit_value_errors = np.sqrt (np.diag(pcov) )
# residuals = ydata[1:] - curve_func(xdata[1:], *popt)
# chi_squared = np.sum((residuals / yerr[1:]) ** 2)

# for j in range(len(popt)):
#     print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
# print(f"\tChi: {chi_squared}")

# fit_vals = np.linspace(0, np.max(xdata)*1.02, 300)

################################################################################################

# plt.figure()

# plt.grid()
# plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt='o', label="Messdaten", color='b', ms=2, zorder=10, alpha=1)
# # plt.plot(fit_vals, curve_func(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared,1)}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)
# plt.legend()

# plt.title("Transmission gegen Absorberdicke")
# plt.xlabel(r"Absorberdicke $d$ / $mm$")
# plt.ylabel(r"Logarithmus der Transmission $\log{T}$")

# plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)