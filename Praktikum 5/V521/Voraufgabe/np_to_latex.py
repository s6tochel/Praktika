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
        for column in columns:
            if round_to != False:
                content = str(np.round(column[i], round_to))
            else:
                content = str(column[i])
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

sorted_file_names = ['spectrum_line1.txt', 'spectrum_line234.txt', 'spectrum_line56.txt']

sorted_file_list = [data_path + f for f in sorted_file_names]

join_files(sorted_file_list, data_path + "big_data.txt")

data = np.loadtxt(data_path + "big_data.txt", delimiter=" ", usecols = range(1, 7)).T

to_latex_table(data, current_path + "latex_table_data",round_to=2)