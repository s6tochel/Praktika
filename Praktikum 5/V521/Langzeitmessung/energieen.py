import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from np_to_latex import to_latex_table

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data = np.loadtxt(dir_path + "data.txt", delimiter=" ").T

b = data[2]
b_err = data[3]

m = 0.09639431861354238
m_err = 2.2835959810843312e-14
n = 0.12575450231051427
n_err = 5.3395536894038764e-08

E = m*b + n
E_err = np.sqrt( (m*b_err)**2 + (b*m_err)**2 + n_err**2 )

to_latex_table([E, E_err], dir_path + "energien.txt", round_to=[3,3])