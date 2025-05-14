import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from np_to_latex import to_latex_table

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

hpge_data_Untergrund = np.loadtxt(dir_path + "Messdaten/" + "Untergrund_HPGe.txt", delimiter=" ", dtype=int).T
hpge_data_Cs = np.loadtxt(dir_path + "Messdaten/" + "Cs_HPGe.txt", delimiter=" ", dtype=int).T
hpge_data_Co = np.loadtxt(dir_path + "Messdaten/" + "Co_HPGe.txt", delimiter=" ", dtype=int).T
hpge_data_Eu = np.loadtxt(dir_path + "Messdaten/" + "Eu_HPGe.txt", delimiter=" ", dtype=int).T
nai_data_untergrund = np.loadtxt(dir_path + "Messdaten/" + "Untergrund_NaI.txt", delimiter=" ", dtype=int).T
nai_data_Cs = np.loadtxt(dir_path + "Messdaten/" + "Cs_NaI.txt", delimiter=" ", dtype=int).T
nai_data_Co = np.loadtxt(dir_path + "Messdaten/" + "Co_NaI.txt", delimiter=" ", dtype=int).T


hpge_Cs = hpge_data_Cs[1] - hpge_data_Untergrund[1]
hpge_Co = hpge_data_Co[1] - hpge_data_Untergrund[1]
Eu = hpge_data_Eu[1] - hpge_data_Untergrund[1]
nai_Cs = nai_data_Cs[1] - hpge_data_Untergrund[1]
nai_Co = nai_data_Co[1] - hpge_data_Untergrund[1]

hpge_Cs_tot = np.sum(hpge_Cs)
hpge_Co_tot = np.sum(hpge_Co)
nai_Cs_tot = np.sum(nai_Cs)
nai_Co_tot = np.sum(nai_Co)

hpge_Cs_peak = np.sum(hpge_Cs[6863-3*7:6863+1+3*7])
hpge_Co_peak1 = np.sum(hpge_Co[12170-3*8:12170+1+3*8])
hpge_Co_peak2 = np.sum(hpge_Co[13823-3*8:13823+1+3*8])
nai_Cs_peak = np.sum(nai_Cs[5600-3*165:5600+1+3*165])
nai_Co_peak1 = np.sum(nai_Co[9711-3*211:9711+1+3*211])
nai_Co_peak2 = np.sum(nai_Co[11011-3*249:11011+1+3*249])

print(hpge_Co[13823-3*8:13823+1+3*8])


Eu_peak1 = np.sum(Eu[1262-3*5:1262+1+3*5])
Eu_peak2 = np.sum(Eu[2537-3*5:2537+1+3*5])
Eu_peak3 = np.sum(Eu[3570-3*6:3570+1+3*6])
Eu_peak7 = np.sum(Eu[8080-3*7:8080+1+3*7])
Eu_peak8 = np.sum(Eu[8997-3*7:8997+1+3*7])
Eu_peak9 = np.sum(Eu[10000-3*8:10000+1+3*8])
Eu_peak10 = np.sum(Eu[11264-3*8:11264+1+3*8])
Eu_peak12 = np.sum(Eu[11536-3*8:11536+1+3*8])
Eu_peak13 = np.sum(Eu[14606-3*9:14606+1+3*9])

total = np.array([hpge_Cs_tot, hpge_Co_tot, nai_Cs_tot, nai_Co_tot])
peaks = np.array([hpge_Cs_peak, hpge_Co_peak1+hpge_Co_peak2, nai_Cs_peak, nai_Co_peak1+nai_Co_peak2])
Eu_peaks = np.array([Eu_peak1, Eu_peak2, Eu_peak3, Eu_peak7, Eu_peak8, Eu_peak9, Eu_peak10, Eu_peak12, Eu_peak13])

print(total)

to_latex_table(
    [total, np.sqrt(total), peaks, np.sqrt(peaks), peaks/total, (peaks/total)*np.sqrt((1/peaks) + (1/total))],
    dir_path + "ppt.txt",
    round_to=[1, 1, 1, 1, 4, 4]
               )

E = np.array([121.7828343, 244.68766158, 344.27180967, 778.93568111, 867.36660395, 964.09834685, 1085.89298118, 1112.10118512, 1408.02845613])
E_err = np.array([6.00304465e-05, 3.93680392e-04, 1.37707565e-04, 9.37920019e-04, 4.70693823e-03, 1.15629976e-03, 1.91873986e-03, 1.59615595e-03, 1.24216853e-03])

I_rel = np.array([28.54, 7.55, 26.59, 12.93, 4.23, 14.51, 10.11, 13.67, 20.87]) * 0.01
I_rel_err = np.array([0.16, 0.04, 0.2, 0.08, 0.03, 0.07, 0.05, 0.08, 0.09])*0.01

A_Cs = 405000
A_Eu = 709000
t = 300
d_hpge = 55.7
d_nai = 50.8
r_hpge_Cs = 100
r_hpge_Eu = 150
r_nai_Cs = 100
r_nai_Eu = 200
r_hpge_err = 5
r_nai_err = 1

P_nai_Cs = (d_nai / ( 4 * r_nai_Cs )) ** 2
P_hpge_Cs = (d_hpge / ( 4 * r_hpge_Cs )) ** 2
P_hpge_Eu = (d_hpge / ( 4 * r_hpge_Eu )) ** 2
P_nai_Cs_err = d_nai**2 * r_nai_err / (8 * r_nai_Cs**3)
P_hpge_Cs_err = d_hpge**2 * r_hpge_err / (8 * r_hpge_Cs**3)
P_hpge_Eu_err = d_hpge**2 * r_hpge_err / (8 * r_hpge_Eu**3)

N_nai_Cs = A_Cs * t * P_nai_Cs
N_hpge_Cs = A_Cs * t * P_hpge_Cs
N_hpge_Eu = A_Eu * t * P_hpge_Eu * I_rel
N_nai_Cs_err = N_nai_Cs * (P_nai_Cs_err / P_nai_Cs)
N_hpge_Cs_err = N_hpge_Cs * (P_hpge_Cs_err / P_hpge_Cs)
N_hpge_Eu_err = N_hpge_Eu * np.sqrt ( (P_hpge_Eu_err / P_hpge_Eu)**2 + (I_rel_err / I_rel)**2 )

eff_nai_Cs = nai_Cs_peak / N_nai_Cs
eff_hpge_Cs = hpge_Cs_peak / N_hpge_Cs
eff_hpge_Eu = Eu_peaks / N_hpge_Eu
eff_nai_Cs_err = eff_nai_Cs * np.sqrt( (1/nai_Cs_peak) + (N_nai_Cs_err / N_nai_Cs)**2 )
eff_hpge_Cs_err = eff_hpge_Cs * np.sqrt( (1/hpge_Cs_peak) + (N_hpge_Cs_err / N_hpge_Cs)**2 )
eff_hpge_Eu_err = eff_hpge_Eu * np.sqrt( (1/Eu_peaks) + (N_hpge_Eu_err / N_hpge_Eu)**2 )

print(f"Effizienz NaI Cs: \t{eff_nai_Cs} \t± {eff_nai_Cs_err}")
print(f"Effizienz HPGe Cs: \t{eff_hpge_Cs} \t± {eff_hpge_Cs_err}")

to_latex_table(
    [E, Eu_peaks, np.sqrt(Eu_peaks), eff_hpge_Eu*100, eff_hpge_Eu_err*100],
    dir_path + "Eu_N_eff.txt",
    round_to= [0, 1, 0, 3, 3]
               )

to_latex_table(
    [E, eff_hpge_Eu, eff_hpge_Eu_err],
    dir_path + "Eu_eff.txt",
    round_to= [0, 3, 6]
               )

def linear(x, a, b):
    return a*x + b

bounds_i = [0, 0, -1]
bounds_i = [np.inf, np.inf, 1]

x_data = E
y_data = np.log(eff_hpge_Eu)
x_err = E_err
y_err = eff_hpge_Eu_err/eff_hpge_Eu

popt, pcov = curve_fit(f=linear, xdata=x_data, ydata=y_data, sigma=y_err, absolute_sigma=True)
fit_values = popt
fit_value_errors = np.diag(pcov)
residuals = ydata=y_data - linear(x_data, *popt)
chi_squared = np.sum((residuals / y_err) ** 2)

fit_vals = np.linspace(0, 1500, 300)
plt.plot(fit_vals, linear(fit_vals, *popt), label=r"linearer Fit ($\chi^2 \approx $" + f"{np.round(chi_squared, 0)}"[:-2] + ")", color="black", linewidth=1, zorder=3, alpha=1)

for j in range(2):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")

Cs_E = np.array([661.68817])
Cs_E_err = np.array([6e-05])

plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', label='Messwerte der Eu-Quelle', color='b', ms=3, zorder=4, alpha=0.8)
plt.errorbar(Cs_E, np.array([np.log(eff_hpge_Cs)]), xerr=Cs_E_err, yerr=np.array([eff_hpge_Cs_err/eff_hpge_Cs]), fmt='o', label='Messwert der Cs-Quelle', color='g', ms=3, zorder=4, alpha=0.8)

plt.legend()
plt.grid()
plt.xlabel(r"Photonenenergie $E_\gamma$ / keV")
plt.ylabel(r"Logarithmus der Effizienz log($\epsilon$)")

plt.title("Fit zur Bestimmung der Energieabhänigkeit der Effizienz")

plt.savefig(dir_path + "figure1.png", dpi=300)

print(N_nai_Cs)
print(N_nai_Cs_err)
print(N_hpge_Cs)
print(N_hpge_Cs_err)
print(hpge_Co_peak1 + hpge_Co_peak2)