import numpy as np

def check_list_same_lenght(list):
    lenght = len(list[0])
    for array in list:
        if len(array) != lenght:
            return False
    return True

def to_latex_table(columns, file_name="latex_table_content", point_to_comma=True):

    if not check_list_same_lenght:
        raise IndexError

    data_lenght = len(columns[0])
    Lines = []

    for i in range(data_lenght):
        line = "         "
        for column in columns:
            content = str(column[i])
            if point_to_comma:
                content = content.replace('.', ',')
            line = line + f"{content} & "
        line = line[:-2] + "\\\\ \hline\n"
        Lines.append(line)

    file = open("latex_table_content", "w+")
    file.writelines(Lines) 
    file.close()