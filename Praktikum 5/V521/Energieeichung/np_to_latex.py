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