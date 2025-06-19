import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from np_to_latex import to_latex_table

filename_filter = "Dosimetrie2"

current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
parent_dir_path = os.path.abspath(os.path.join(current_path, os.pardir)) + "/"
data_path = parent_dir_path + "Daten/Data_Files/"
table_path = current_path + "LatexTabellen/"

filelist = [filename for filename in os.listdir(data_path) if  filename_filter in filename]
namedict = {filename : filename[len(filename_filter):-4] for filename in filelist}

datadict = {namedict[filename] : np.loadtxt(data_path + filename, delimiter=",").T for filename in filelist}


################################################################################################

selector = "b"

round_to = [1, 2, 2]

data1 = datadict[selector]

to_latex_table([data1[0], data1[1], data1[2]], table_path+filename_filter+selector+".txt", point_to_comma=True, round_to=round_to)