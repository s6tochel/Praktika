import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

filename_filter = "Dosimetrie1"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
figure_path = current_path + "Abbildungen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}


################################################################################################

data1 = datadict["d"]
data2 = datadict["e"]
data3 = datadict["f"]

filename = "Cu"

################################################################################################

plt.figure()
plt.grid()
plt.errorbar(data1[0], data1[1], xerr=data1[0]*0.01, yerr=data1[2], fmt='o', label="Cu U=15kV", color='b', ms=2, zorder=10, alpha=1)
plt.errorbar(data2[0], data2[1], xerr=data2[0]*0.01, yerr=data2[2], fmt='o', label="Cu U=25kV", color='r', ms=2, zorder=9, alpha=1)
plt.errorbar(data3[0], data3[1], xerr=data3[0]*0.01, yerr=data3[2], fmt='o', label="Cu U=10kV", color='g', ms=2, zorder=8, alpha=1)
plt.title("Ionisationsstrom gegen Kondensatorspannung")
plt.xlabel(r"Kondensatorspannung $U_c$ / V")
plt.ylabel(r"Ionisationsstrom $I_c$ / nA")
plt.legend()
plt.savefig(figure_path + filename_filter + filename + ".png", dpi=300)