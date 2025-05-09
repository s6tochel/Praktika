import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from np_to_latex import to_latex_table

def linear(x, a,b):
    return a*x +b

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data_hpge = np.loadtxt(dir_path + "data_hpge.txt", delimiter=" ").T
data_nai = np.loadtxt(dir_path + "data_nai.txt", delimiter=" ").T

b_hpge = data_hpge[2]
b_hpge_err = data_hpge[3]
b_nai = data_nai[2]
b_nai_err = data_nai[3]
sigma_hpge = data_hpge[4]
sigma_hpge_err = data_hpge[5]
sigma_nai = data_nai[4]
sigma_nai_err = data_nai[5]

m_nai_old = 8.263947995069694
m_nai_err_old = 1.456387810485596e-07
n_nai_old = 124.20649170283689
n_nai_err_old = 0.01919811433712138
m_hpge_old = 10.374055384001757
m_hpge_err_old = 2.457629404221388e-12
n_hpge_old = -1.304584171756852
n_hpge_err_old = 5.539282569971812e-07

m_nai = 1 / m_nai_old
m_nai_err = m_nai * (m_nai_err_old / m_nai_old)
n_nai = - n_nai_old / m_nai_old
n_nai_err = n_nai * np.sqrt( (n_nai_err_old / n_nai_old)**2 + (m_nai_err_old / m_nai_old)**2 )
m_hpge = 1 / m_hpge_old
m_hpge_err = m_hpge * (m_hpge_err_old / m_hpge_old)
n_hpge = - n_hpge_old / m_hpge_old
n_hpge_err = n_hpge * np.sqrt( (n_hpge_err_old / n_hpge_old)**2 + (m_hpge_err_old / m_hpge_old)**2 )

E_nai = m_nai * b_nai + n_nai
E_nai_err = np.sqrt( ((m_nai*b_nai)**2)*((m_nai_err/m_nai)**2 + (b_nai_err/b_nai)**2) + n_nai_err**2 )
E_hpge = m_hpge * b_hpge + n_hpge
E_hpge_err = np.sqrt( ((m_hpge*b_hpge)**2)*((m_hpge_err/m_hpge)**2 + (b_hpge_err/b_hpge)**2) + n_hpge_err**2 )

j = 2 * np.sqrt(2 * np.log(2))
FWHM_nai = m_nai * (j * sigma_nai) + n_nai
FWHM_nai_err = np.sqrt( ((m_nai*(j * sigma_nai))**2)*((m_nai_err/m_nai)**2 + (sigma_nai_err/sigma_nai)**2) + n_nai_err**2 )
FWHM_hpge = m_hpge * (j * sigma_hpge) + n_hpge
FWHM_hpge_err = np.sqrt( ((m_hpge*(j * sigma_hpge))**2)*((m_hpge_err/m_hpge)**2 + (sigma_hpge_err/sigma_hpge)**2) + n_hpge_err**2 )

slice = list(range(3, len(E_hpge)))

x = E_hpge
y = FWHM_hpge**2
x_err = E_hpge_err
y_err = y * 2 * FWHM_hpge_err / FWHM_hpge

to_latex_table([E_nai, E_nai_err, FWHM_nai, FWHM_nai_err], dir_path + "nai_energy_fwhm.txt", round_to=5)
to_latex_table([E_hpge, E_hpge_err, FWHM_hpge, FWHM_hpge_err, y, y_err], dir_path + "hpge_energy_fwhm.txt", round_to=5)

popt, pcov = curve_fit(f=linear, xdata=x[slice], ydata=y[slice], sigma=y_err[slice], absolute_sigma=True)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = y[slice] - linear(x[slice], *popt)
chi_squared = np.sum((residuals / y_err[slice]) ** 2)

for j in range(2):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

fit_vals = np.linspace(100, 1500, 300)
plt.plot(fit_vals, linear(fit_vals, *popt), label=r"Linearer Fit ($\chi^2 \approx $" + f"{int(np.round(chi_squared,0))}" + ")", color="black", linewidth=1, zorder=3, alpha=0.8)

plt.xlabel(r"Photonenenergie $E_\gamma$ / keV")
plt.ylabel("quadrierte Halbwertsbreite $\Delta E ^2$ / keV")
plt.title("Geradenfit Bestimmung zur Halbwertszeitkomposition")

plt.errorbar(x[slice], y[slice], yerr=y_err[slice], xerr=x_err[slice], fmt='o', label=f'Messwerte', color='b', ms=4, zorder=4, alpha=1)

plt.grid()
plt.legend()

plt.savefig(dir_path + "figure1.png", dpi=300)

print("\nNaI:")
print(f"m':\t{m_nai}\t±  {m_nai_err}")
print(f"n':\t{n_nai}\t±  {n_nai_err}\n")
print("HPGe:")
print(f"m':\t{m_hpge}\t±  {m_hpge_err}")
print(f"n':\t{n_hpge}\t±  {n_hpge_err}\n")