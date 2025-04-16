import numpy as np

def readin(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    # del lines[0:5]

    array_index = np.array([])
    array_data = np.array([])

    for i in range(len(lines)):
        array_index = np.append(array_index, np.array([i]))
        value = int(''.join(i for i in lines[i] if i.isdigit()))
        array_data = np.append(array_data, np.array([value]))
    
    return np.vstack((array_index, array_data))

def bound_maker(gauss_fits, H_is, H_os, A_is, A_os, x0_is, x0_os, sigma_is, sigma_os):
    single_i_peaks = []
    single_o_peaks = []
    Master_list = []
    for i in range(len(A_is)):
        single_i_peaks.append([A_is[i], x0_is[i], sigma_is[i]])
        single_o_peaks.append([A_os[i], x0_os[i], sigma_os[i]])
    for i in range(len(gauss_fits)):
        params_i = [H_is.pop(0)]
        params_o = [H_os.pop(0)]
        for j in range(gauss_fits[i]):
            params_i = params_i + single_i_peaks.pop(0)
            params_o = params_o + single_o_peaks.pop(0)
        Master_list.append((np.array(params_i), np.array(params_o)))
    return Master_list