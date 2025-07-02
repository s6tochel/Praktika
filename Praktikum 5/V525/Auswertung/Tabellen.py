import numpy as np
from np_to_latex import to_latex_table
import os

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = current_path + "Fit_data/"

data1 = np.loadtxt(data_path + "Spektrum6.txt", delimiter=" ", dtype=float)
data2 = np.loadtxt(data_path + "Spektrum7_a.txt", delimiter=" ", dtype=float)
data3 = np.loadtxt(data_path + "Spektrum7_b.txt", delimiter=" ", dtype=float)

data = np.vstack((data1, data2, data3)).T

to_latex_table([data[0], data[3], data[5], data[6]], data_path + "Line_data_links.txt", round_to=[0,0,1,1])