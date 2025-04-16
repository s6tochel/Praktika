import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

filename = "spectrum.txt"

def save(filename):
    plt.savefig(dir_path + "/figures/" + filename + ".png", dpi=300)

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = dir_path + "/" + filename

data = np.loadtxt(file_path, delimiter="\t", dtype=int).T

plt.scatter(data[0], data[1], s=2)
save("spectrum")