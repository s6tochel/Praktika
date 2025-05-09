import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_boden = np.loadtxt(dir_path + "/Messdaten/" + "Langzeit_Bodenprobe.txt", delimiter=" ", dtype=int).T
data_unter = np.loadtxt(dir_path + "/Messdaten/" + "Langzeit_Untergrund.txt", delimiter=" ", dtype=int).T

data_diff = data_boden[1] - data_unter[1]
data_err = np.sqrt(data_boden[1] + data_unter[1])

data_file_dir = dir_path + "/Messdaten/" + "Differenz" + ".txt"
data_file = open(data_file_dir, "w")

print(data_boden[0])

for i in range(len(data_diff)):
    data_file.write(f"{data_boden[0][i]} {data_diff[i]} {data_err[i]}\n")

data_file.close