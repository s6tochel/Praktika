import os
from os.path import isfile, join
import numpy as np

def check_list_same_lenght(list):
    lenght = len(list[0])
    for array in list:
        if len(array) != lenght:
            return False
    return True

def to_latex_table(columns, file_name="latex_table_content", point_to_comma=True, round_to=False):

    if not check_list_same_lenght:
        raise IndexError

    data_lenght = len(columns[0])
    Lines = []

    for i in range(data_lenght):
        line = "         "
        for j in range(len(columns)):
            if round_to == False:
                content = str(columns[j][i])
            elif isinstance(round_to, float) or isinstance(round_to, int):
                content = str(np.round(columns[j][i], round_to))
                if round_to <= 0:
                    content = content[:-2]
            elif isinstance(round_to, list) and len(round_to) == len(columns):
                content = str(np.round(columns[j][i], round_to[j]))
                if round_to[j]<= 0:
                    content = content[:-2]
            else:
                print("round_to has to be set to False, be an int/float or be a list of same lenght as columns")
                raise TypeError
            if point_to_comma:
                content = content.replace('.', ',')
            line = line + f"{content} & "
        line = line[:-2] + "\\\\ \hline\n"
        Lines.append(line)

    file = open(file_name, "w")
    file.writelines(Lines) 
    file.close()

def join_files(list_of_files, new_file_name):
    entire_text = ""
    for file_name in list_of_files:
        file = open(file_name, "r")
        content = file.read()
        entire_text = entire_text + content
        file.close()
    new_file = open(new_file_name, "w")
    new_file.write(entire_text)
    new_file.close()



current_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data_path = current_path + "line_data/"

file_list = [data_path + f for f in os.listdir(data_path) if isfile(join(data_path, f))]
file_names = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]
print(file_names)

sorted_file_names = ['Untergrund_NaI_fits.txt', 'Cs_NaI_fits.txt', 'Co_NaI_fits.txt', 'Eu_NaI_fits1.txt', 'Eu_NaI_fits2.txt', 'Eu_NaI_fits3.txt', 'Eu_NaI_fits4.txt']

sorted_file_list = [data_path + f for f in sorted_file_names]

join_files(sorted_file_list, data_path + "big_data.txt")

data = np.loadtxt(data_path + "big_data.txt", delimiter=" ", usecols = range(2, 8)).T

data = np.insert(data, 0, np.array(range(data[0].size))+1, axis=0)

rounding_list = [0, 0, 0, 2, 3, 2, 4]

to_latex_table(data, current_path + "latex_table_data",round_to=rounding_list)