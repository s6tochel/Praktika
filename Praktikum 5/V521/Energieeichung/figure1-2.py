import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

E_lit = [661.7, 1173.2, 1332.5, 121.7817, 244.6974, 344.2785, 778.9045, 867.380, 964.057, 1085.837, 1112.076, 1408.013]
E_lit_err = [0, 0, 0, 0.0003, 0.0008, 0.0012, 0.0024, 0.003, 0.005, 0.010, 0.003, 0.003]

dir_path = os.path.dirname(os.path.realpath(__file__))

data_hpge = np.loadtxt(dir_path + "/data_hpge.txt", delimiter=" ").T
data_nai = np.loadtxt(dir_path + "/data_nai.txt", delimiter=" ").T

b_nai = data_nai[2]
b_hpge = data_hpge[2]

b_nai_err = data_nai[3]
b_hpge_err = data_hpge[3]


nai_scale = 0.125
hpge_scale = 0.1
nai_offset = 0

hpge_takeout = [6,7,8, -3]
nai_takeout = [3, 4, 5]

# b_hpge = np.delete(b_hpge, hpge_takeout)
# b_hpge_err = np.delete(b_hpge_err, hpge_takeout)
# b_nai = np.delete(b_nai, nai_takeout)
# b_nai_err = np.delete(b_nai_err, nai_takeout)

fig,ax = plt.subplots(1)

# Make your plot, set your axes labels
ax.scatter(nai_offset + b_nai * nai_scale, np.ones(len(b_nai)) * 2, label="NaI (b * 0,125)")
ax.scatter(b_hpge * hpge_scale, np.ones(len(b_hpge)) * 3, label="HPGe (b * 0,1)")
ax.scatter(E_lit, np.ones(len(E_lit)) * 1, label="Literaturwerte")
ax.set_xlabel(r"Energie $E$ / keV")
ax.legend()
ax.set_ylim(0.5, 4)
plt.title("Vergleich der ermittelten Peaks zu den Literatur-Energiewerten")

# Turn off tick labels
ax.set_yticklabels([])

plt.savefig(dir_path + "/figure1.png", dpi=300)




# plt.scatter(nai_offset + b_nai * nai_scale, np.ones(len(b_nai)), label="NaI (b * 0,125)")
# plt.scatter(b_hpge * hpge_scale, np.ones(len(b_hpge)) * 2, label="HPGe (b * 0,1)")
# plt.scatter(E_lit, np.ones(len(E_lit)) * 3, label="Literatur")
# plt.legend()
# plt.ylim(0.5, 4.5)
# plt.xlabel(r"Energie $E$ / keV")

# plt.savefig(dir_path + "/figure.png", dpi=300)