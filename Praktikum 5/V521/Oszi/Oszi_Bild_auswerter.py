import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

data = np.loadtxt(dir_path + "scope_9.csv", delimiter=",").T
t = data[0] * 1000000
U = data[1] * 1000

#len t = 2000
range = (800, 1600)

t_slice = t[range[0]:range[1]]
U_slice = U[range[0]:range[1]]

plt.scatter(t_slice, U_slice, s=4)
plt.xlabel(r'$t$ / $\mu$s')
plt.ylabel(r'$U$ / mV')
plt.grid()
plt.title("Oszillograph des Vorverstärkersignals mit HPGe-Detektor")
plt.savefig(dir_path + "scope_9.png",  dpi=300)